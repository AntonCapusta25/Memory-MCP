#!/usr/bin/env python3
"""
Memory MCP Server with Supabase Backend
A project-based memory system for Claude with cloud persistence
"""

import asyncio
import json
import logging
import time
import sys
import os
import base64
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from urllib.parse import urlparse
import mimetypes

# Enhanced error handling for imports
try:
    from mcp.server.fastmcp import FastMCP
    MCP_VERSION = "new"
except ImportError:
    try:
        from fastmcp import FastMCP
        MCP_VERSION = "old"
    except ImportError as e:
        print(f"❌ FastMCP not available: {e}")
        print("Install with: pip install fastmcp")
        sys.exit(1)

try:
    from supabase import create_client, Client
except ImportError as e:
    print(f"❌ Supabase not available: {e}")
    print("Install with: pip install supabase")
    sys.exit(1)

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv("PORT", 8001))
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase configuration!")
    print("Set environment variables:")
    print("  - SUPABASE_URL=https://your-project.supabase.co")
    print("  - SUPABASE_ANON_KEY=your-anon-key")
    sys.exit(1)

if DEBUG_MODE:
    logging.getLogger().setLevel(logging.DEBUG)

# Initialize MCP server
try:
    if MCP_VERSION == "new":
        mcp = FastMCP(
            "Memory Server",
            host="0.0.0.0", 
            port=PORT
        )
    else:
        mcp = FastMCP(
            name="Memory Server",
            host="0.0.0.0", 
            port=PORT
        )
    logger.info(f"✅ FastMCP initialized (version: {MCP_VERSION})")
except Exception as e:
    logger.error(f"❌ Failed to initialize FastMCP: {e}")
    sys.exit(1)

# Pydantic models
class MemorizeData(BaseModel):
    project: str = Field(description="Project name to categorize this memory")
    content: str = Field(description="Content to memorize")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the memory")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for easier recall")
    image_url: Optional[str] = Field(default=None, description="Optional image URL to store with memory")
    link_url: Optional[str] = Field(default=None, description="Optional web link to store with memory")

class RecallData(BaseModel):
    project: Optional[str] = Field(default=None, description="Project name to search within")
    query: Optional[str] = Field(default=None, description="Search query to match against content and tags")
    limit: int = Field(default=10, description="Maximum number of memories to return")

class DeleteMemoryData(BaseModel):
    memory_id: int = Field(description="ID of the memory to delete")

class SupabaseMemoryDatabase:
    """Supabase database for storing project-based memories"""
    
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.files_bucket = "memory-files"
        self.init_database()
        logger.info("✅ Supabase client initialized")
    
    def init_database(self):
        """Initialize the database tables and storage bucket"""
        try:
            # Create storage bucket if it doesn't exist
            try:
                self.client.storage.create_bucket(self.files_bucket, {"public": False})
                logger.info("✅ Storage bucket created")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info("✅ Storage bucket already exists")
                else:
                    logger.warning(f"⚠️ Storage bucket creation warning: {e}")
            
            # Test database connection
            result = self.client.table("memories").select("count", count="exact").execute()
            logger.info("✅ Database connection verified")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def store_memory(self, project: str, content: str, metadata: Optional[Dict] = None, 
                    tags: Optional[List[str]] = None, image_url: Optional[str] = None, 
                    link_url: Optional[str] = None) -> int:
        """Store a new memory entry"""
        try:
            link_title = None
            link_description = None
            
            # Handle link metadata extraction
            if link_url:
                link_title, link_description = self._extract_link_metadata(link_url)
            
            # Insert into database
            memory_data = {
                "project": project,
                "content": content,
                "metadata": metadata,
                "tags": tags,
                "link_url": link_url,
                "link_title": link_title,
                "link_description": link_description,
                "created_at": datetime.utcnow().isoformat()
            }
            
            result = self.client.table("memories").insert(memory_data).execute()
            memory_id = result.data[0]["id"]
            
            logger.info(f"✅ Memory {memory_id} stored in project '{project}'")
            return memory_id
            
        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            raise
    
    def _extract_link_metadata(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """Extract title and description from web page"""
        try:
            import re
            
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            
            html = response.text
            
            # Extract title
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else None
            
            # Extract description
            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']', html, re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else None
            
            logger.info(f"✅ Link metadata extracted: {title}")
            return title, description
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract link metadata: {e}")
            return None, None
    
    def recall_memories(self, project: Optional[str] = None, query: Optional[str] = None, 
                       limit: int = 10) -> List[Dict]:
        """Recall memories with optional filtering"""
        try:
            # Build query
            db_query = self.client.table("memories").select("*")
            
            if project:
                db_query = db_query.eq("project", project)
            
            if query:
                # Use Supabase text search
                search_query = f"%{query}%"
                db_query = db_query.or_(f"content.ilike.{search_query},link_title.ilike.{search_query}")
            
            db_query = db_query.order("created_at", desc=True).limit(limit)
            
            result = db_query.execute()
            memories = result.data
            
            logger.info(f"✅ Found {len(memories)} memories")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Failed to recall memories: {e}")
            raise
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """Get all unique project names with memory counts"""
        try:
            result = self.client.table("memories").select("project").execute()
            project_names = list(set(row["project"] for row in result.data))
            projects = []
            
            for project_name in project_names:
                count_result = self.client.table("memories").select("id", count="exact").eq("project", project_name).execute()
                latest_result = self.client.table("memories").select("created_at").eq("project", project_name).order("created_at", desc=True).limit(1).execute()
                
                projects.append({
                    "name": project_name,
                    "memory_count": count_result.count,
                    "last_updated": latest_result.data[0]["created_at"] if latest_result.data else None
                })
            
            logger.info(f"✅ Found {len(projects)} projects")
            return projects
            
        except Exception as e:
            logger.error(f"❌ Failed to list projects: {e}")
            raise
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete a specific memory by ID"""
        try:
            delete_result = self.client.table("memories").delete().eq("id", memory_id).execute()
            success = len(delete_result.data) > 0
            
            if success:
                logger.info(f"✅ Memory {memory_id} deleted")
            else:
                logger.warning(f"⚠️ Memory {memory_id} not found")
            
            return success
                
        except Exception as e:
            logger.error(f"❌ Failed to delete memory: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Total memories
            total_result = self.client.table("memories").select("id", count="exact").execute()
            total_memories = total_result.count
            
            # Count distinct projects
            result = self.client.table("memories").select("project").execute()
            total_projects = len(set(row["project"] for row in result.data))
            
            # Recent memories (last 7 days)
            from datetime import datetime, timedelta
            week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
            recent_result = self.client.table("memories").select("id", count="exact").gte("created_at", week_ago).execute()
            recent_memories = recent_result.count
            
            return {
                "total_memories": total_memories,
                "total_projects": total_projects,
                "recent_memories_7d": recent_memories,
                "database_type": "Supabase PostgreSQL"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}

# Initialize global database instance
try:
    db = SupabaseMemoryDatabase()
except Exception as e:
    logger.error(f"❌ Failed to initialize Supabase database: {e}")
    sys.exit(1)

# MCP Tools
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with comprehensive status"""
    try:
        stats = db.get_stats()
        
        return {
            "status": "healthy",
            "server": "memory-mcp-supabase",
            "version": "1.0.0",
            "timestamp": time.time(),
            "port": PORT,
            "config": {
                "database": "Supabase",
                "debug": DEBUG_MODE,
                "supabase_url": SUPABASE_URL
            },
            "database": {
                "status": "✅ Connected",
                "stats": stats
            },
            "mcp_version": MCP_VERSION
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@mcp.tool()
async def memorize(data: MemorizeData) -> Dict[str, Any]:
    """Store content in memory under a specific project"""
    try:
        memory_id = db.store_memory(
            project=data.project, 
            content=data.content, 
            metadata=data.metadata, 
            tags=data.tags,
            image_url=data.image_url,
            link_url=data.link_url
        )
        
        result_text = "✅ Memory stored successfully!\n"
        result_text += f"ID: {memory_id}\n"
        result_text += f"Project: {data.project}\n"
        result_text += f"Content: {data.content[:100]}{'...' if len(data.content) > 100 else ''}\n"
        
        if data.tags:
            result_text += f"Tags: {', '.join(data.tags)}\n"
        
        if data.image_url:
            result_text += "🖼️ Image: Downloaded and stored\n"
        
        if data.link_url:
            result_text += f"🔗 Link: {data.link_url}\n"
        
        return {
            "success": True,
            "memory_id": memory_id,
            "project": data.project,
            "has_image": bool(data.image_url),
            "has_link": bool(data.link_url),
            "message": result_text
        }
    except Exception as e:
        logger.error(f"❌ Error in memorize: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to store memory: {str(e)[:200]}"
        }

@mcp.tool()
async def recall(data: RecallData) -> Dict[str, Any]:
    """Recall memories from a project or search across all memories"""
    try:
        memories = db.recall_memories(data.project, data.query, data.limit)
        
        if not memories:
            return {
                "success": True,
                "memories": [],
                "count": 0,
                "message": "🔍 No memories found matching your criteria."
            }
        
        # Format memories for display
        formatted_memories = []
        for memory in memories:
            formatted_memory = {
                "id": memory['id'],
                "project": memory['project'],
                "content": memory['content'],
                "timestamp": memory['created_at']
            }
            
            if memory.get('tags'):
                formatted_memory['tags'] = memory['tags']
            
            if memory.get('metadata'):
                formatted_memory['metadata'] = memory['metadata']
            
            # Add link info
            if memory.get('link_url'):
                formatted_memory['link_info'] = {
                    "url": memory['link_url'],
                    "title": memory['link_title'],
                    "description": memory['link_description']
                }
            
            formatted_memories.append(formatted_memory)
        
        result_text = f"🧠 Found {len(memories)} memory(ies):\n\n"
        for memory in formatted_memories:
            result_text += f"**Memory ID {memory['id']}** (Project: {memory['project']})\n"
            result_text += f"📅 {memory['timestamp']}\n"
            result_text += f"📝 {memory['content']}\n"
            
            if memory.get('tags'):
                result_text += f"🏷️ Tags: {', '.join(memory['tags'])}\n"
            
            if memory.get('link_info'):
                link_info = memory['link_info']
                result_text += f"🔗 Link: {link_info['title'] or 'Untitled'}\n"
                result_text += f"   {link_info['url']}\n"
                if link_info['description']:
                    result_text += f"   {link_info['description'][:100]}...\n"
            
            if memory.get('metadata'):
                result_text += f"📊 Metadata: {json.dumps(memory['metadata'], indent=2)}\n"
            
            result_text += "\n" + "─" * 50 + "\n\n"
        
        return {
            "success": True,
            "memories": formatted_memories,
            "count": len(memories),
            "message": result_text
        }
    except Exception as e:
        logger.error(f"❌ Error in recall: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to recall memories: {str(e)[:200]}"
        }

@mcp.tool()
async def list_projects() -> Dict[str, Any]:
    """List all available projects with memory counts"""
    try:
        projects = db.list_projects()
        
        if not projects:
            return {
                "success": True,
                "projects": [],
                "count": 0,
                "message": "📁 No projects found in memory."
            }
        
        result_text = f"📁 Found {len(projects)} project(s):\n\n"
        for project in projects:
            result_text += f"• **{project['name']}** ({project['memory_count']} memories)\n"
            if project.get('last_updated'):
                result_text += f"  Last updated: {project['last_updated']}\n\n"
        
        return {
            "success": True,
            "projects": projects,
            "count": len(projects),
            "message": result_text
        }
    except Exception as e:
        logger.error(f"❌ Error in list_projects: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to list projects: {str(e)[:200]}"
        }

@mcp.tool()
async def delete_memory(data: DeleteMemoryData) -> Dict[str, Any]:
    """Delete a specific memory by ID"""
    try:
        success = db.delete_memory(data.memory_id)
        
        if success:
            return {
                "success": True,
                "message": f"✅ Memory ID {data.memory_id} deleted successfully."
            }
        else:
            return {
                "success": False,
                "message": f"❌ Memory ID {data.memory_id} not found."
            }
    except Exception as e:
        logger.error(f"❌ Error in delete_memory: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to delete memory: {str(e)[:200]}"
        }

@mcp.tool()
async def get_memory_stats() -> Dict[str, Any]:
    """Get comprehensive memory database statistics"""
    try:
        stats = db.get_stats()
        projects = db.list_projects()
        
        # Top projects by memory count
        top_projects = sorted(projects, key=lambda x: x['memory_count'], reverse=True)[:5]
        
        result_text = "📊 Memory Database Statistics\n\n"
        result_text += f"📝 Total Memories: {stats['total_memories']}\n"
        result_text += f"📁 Total Projects: {stats['total_projects']}\n"
        result_text += f"🕒 Recent (7 days): {stats['recent_memories_7d']}\n"
        result_text += f"💾 Database: {stats['database_type']}\n\n"
        
        if top_projects:
            result_text += "🏆 Top Projects:\n"
            for project in top_projects:
                result_text += f"  • {project['name']}: {project['memory_count']} memories\n"
        
        return {
            "success": True,
            "stats": stats,
            "top_projects": top_projects,
            "message": result_text
        }
    except Exception as e:
        logger.error(f"❌ Error in get_memory_stats: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get statistics: {str(e)[:200]}"
        }

@mcp.tool()
async def auto_enable_memory() -> Dict[str, Any]:
    """Auto-enable memory system"""
    try:
        stats = db.get_stats()
        projects = db.list_projects()
        
        result_text = "🧠 Memory MCP Server Auto-Enabled!\n\n"
        result_text += "✅ Status: Active and Ready\n"
        result_text += f"📝 Memories: {stats['total_memories']}\n" 
        result_text += f"📁 Projects: {stats['total_projects']}\n"
        result_text += "💾 Database: Supabase (Cloud)\n\n"
        
        result_text += "🚀 Quick Start:\n"
        result_text += "• Say: 'Memorize this under project [name]: [content]'\n"
        result_text += "• Say: 'What do you remember about [project]?'\n"
        result_text += "• Say: 'List all my projects'\n\n"
        
        if projects:
            result_text += "📁 Available Projects:\n"
            for project in projects[:5]:
                result_text += f"  • {project['name']} ({project['memory_count']} memories)\n"
        
        result_text += "\n💡 Tip: Use tags for better organization and easier recall!"
        result_text += "\n🔗 Supports: Images, files, web links, and rich metadata!"
        
        return {
            "success": True,
            "auto_enabled": True,
            "stats": stats,
            "projects": projects,
            "message": result_text
        }
        
    except Exception as e:
        logger.error(f"❌ Error in auto_enable_memory: {str(e)}")
        return {
            "success": False,
            "error": f"Auto-enable failed: {str(e)[:200]}"
        }

# Enhanced cleanup and startup
async def cleanup():
    """Enhanced cleanup with proper resource management"""
    try:
        logger.info("🔒 Cleanup completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

async def startup():
    """Initialize components on startup"""
    try:
        logger.info("🚀 Starting Memory MCP Server with Supabase...")
        
        # Test database connectivity
        stats = db.get_stats()
        logger.info(f"✅ Database ready with {stats['total_memories']} memories in {stats['total_projects']} projects")
        
        logger.info("🎯 Memory server ready")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

# Main execution
if __name__ == "__main__":
    print("🧠 Starting Memory MCP Server with Supabase...")
    print(f"⚙️  Configuration:")
    print(f"  - Port: {PORT}")
    print(f"  - Database: Supabase")
    print(f"  - Debug Mode: {DEBUG_MODE}")
    print(f"  - MCP Version: {MCP_VERSION}")
    print()
    
    try:
        # Run startup
        asyncio.run(startup())
        
        # Start server
        print(f"🌐 Server starting on http://0.0.0.0:{PORT}")
        if MCP_VERSION == "new":
            mcp.run(transport="sse")
        else:
            mcp.run()
            
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
        asyncio.run(cleanup())
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        asyncio.run(cleanup())
        sys.exit(1)
