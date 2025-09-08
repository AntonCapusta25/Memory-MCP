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

class MemorizeWithFileData(BaseModel):
    project: str = Field(description="Project name to categorize this memory")
    content: str = Field(description="Content to memorize")
    file_data: str = Field(description="Base64 encoded file data")
    file_name: str = Field(description="Original filename")
    file_type: str = Field(description="MIME type of the file")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the memory")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for easier recall")

class RecallData(BaseModel):
    project: Optional[str] = Field(default=None, description="Project name to search within")
    query: Optional[str] = Field(default=None, description="Search query to match against content and tags")
    limit: int = Field(default=10, description="Maximum number of memories to return")
    include_files: bool = Field(default=True, description="Include file attachments in results")

class DeleteMemoryData(BaseModel):
    memory_id: int = Field(description="ID of the memory to delete")


class SupabaseMemoryDatabase:
    """Supabase database for storing project-based memories with file support"""
    
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
            
            # The table should be created via Supabase dashboard or migration
            # This is just a connectivity test
            result = self.client.table("memories").select("count", count="exact").execute()
            logger.info("✅ Database connection verified")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def store_memory(self, project: str, content: str, metadata: Optional[Dict] = None, 
                    tags: Optional[List[str]] = None, file_data: Optional[str] = None,
                    file_name: Optional[str] = None, file_type: Optional[str] = None,
                    image_url: Optional[str] = None, link_url: Optional[str] = None) -> int:
        """Store a new memory entry with optional file and link data"""
        try:
            file_path = None
            file_size = None
            link_title = None
            link_description = None
            
            # Handle file storage
            if file_data and file_name:
                file_path, file_size = self._save_file_to_supabase(file_data, file_name, file_type)
            
            # Handle image URL download
            elif image_url:
                file_path, file_name, file_type, file_size = self._download_and_store_image(image_url)
            
            # Handle link metadata extraction
            if link_url:
                link_title, link_description = self._extract_link_metadata(link_url)
            
            # Insert into database
            memory_data = {
                "project": project,
                "content": content,
                "metadata": metadata,
                "tags": tags,
                "file_path": file_path,
                "file_name": file_name,
                "file_type": file_type,
                "file_size": file_size,
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
    
    def _save_file_to_supabase(self, file_data: str, file_name: str, file_type: str) -> tuple[str, int]:
        """Save base64 file data to Supabase storage"""
        try:
            # Create unique filename
            file_hash = hashlib.md5(file_data.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in file_name if c.isalnum() or c in "._-")
            unique_filename = f"{timestamp}_{file_hash}_{safe_name}"
            
            # Decode file data
            file_bytes = base64.b64decode(file_data)
            file_size = len(file_bytes)
            
            # Upload to Supabase storage
            self.client.storage.from_(self.files_bucket).upload(
                path=unique_filename,
                file=file_bytes,
                file_options={"content-type": file_type}
            )
            
            logger.info(f"✅ File uploaded to Supabase: {unique_filename}")
            return unique_filename, file_size
            
        except Exception as e:
            logger.error(f"❌ Failed to save file to Supabase: {e}")
            raise
    
    def _download_and_store_image(self, image_url: str) -> tuple[str, str, str, int]:
        """Download image from URL and store in Supabase"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Determine file type
            content_type = response.headers.get('content-type', 'image/jpeg')
            ext = mimetypes.guess_extension(content_type) or '.jpg'
            
            # Create filename
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}_{url_hash}{ext}"
            
            # Upload to Supabase storage
            self.client.storage.from_(self.files_bucket).upload(
                path=filename,
                file=response.content,
                file_options={"content-type": content_type}
            )
            
            logger.info(f"✅ Image downloaded and stored: {filename}")
            return filename, filename, content_type, len(response.content)
            
        except Exception as e:
            logger.error(f"❌ Failed to download and store image: {e}")
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
                       limit: int = 10, include_files: bool = True) -> List[Dict]:
        """Recall memories with optional filtering and file support"""
        try:
            # Build query
            db_query = self.client.table("memories").select("*")
            
            if project:
                db_query = db_query.eq("project", project)
            
            if query:
                # Use Supabase text search across multiple columns
                search_query = f"%{query}%"
                db_query = db_query.or_(f"content.ilike.{search_query},tags.cs.{{{query}}},link_title.ilike.{search_query}")
            
            db_query = db_query.order("created_at", desc=True).limit(limit)
            
            result = db_query.execute()
            memories = result.data
            
            # Add file data if requested
            if include_files:
                for memory in memories:
                    if memory.get('file_path'):
                        try:
                            file_data = self._get_file_from_supabase(memory['file_path'])
                            memory['has_file'] = True
                            memory['file_data_base64'] = file_data
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load file {memory['file_path']}: {e}")
                            memory['has_file'] = False
                            memory['file_data_base64'] = None
                    else:
                        memory['has_file'] = False
                        memory['file_data_base64'] = None
            
            logger.info(f"✅ Found {len(memories)} memories")
            return memories
            
        except Exception as e:
            logger.error(f"❌ Failed to recall memories: {e}")
            raise
    
    def _get_file_from_supabase(self, file_path: str) -> Optional[str]:
        """Get file content as base64 string from Supabase storage"""
        try:
            response = self.client.storage.from_(self.files_bucket).download(file_path)
            return base64.b64encode(response).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Failed to download file from Supabase: {e}")
            return None
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """Get all unique project names with memory counts"""
        try:
            # Use Supabase aggregation
            result = self.client.rpc("get_project_stats").execute()
            projects = result.data or []
            
            # Fallback if RPC function doesn't exist
            if not projects:
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
            # Get memory info first to delete associated file
            result = self.client.table("memories").select("file_path").eq("id", memory_id).execute()
            
            if result.data:
                memory = result.data[0]
                
                # Delete associated file from storage
                if memory.get("file_path"):
                    try:
                        self.client.storage.from_(self.files_bucket).remove([memory["file_path"]])
                        logger.info(f"✅ File deleted from storage: {memory['file_path']}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete file: {e}")
                
                # Delete memory record
                delete_result = self.client.table("memories").delete().eq("id", memory_id).execute()
                success = len(delete_result.data) > 0
                
                if success:
                    logger.info(f"✅ Memory {memory_id} deleted")
                else:
                    logger.warning(f"⚠️ Memory {memory_id} not found")
                
                return success
            else:
                logger.warning(f"⚠️ Memory {memory_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete memory: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Total memories
            total_result = self.client.table("memories").select("id", count="exact").execute()
            total_memories = total_result.count
            
            # Total projects
            projects_result = self.client.rpc("count_distinct_projects").execute()
            total_projects = projects_result.data if projects_result.data else 0
            
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

# MCP Tools using FastMCP decorators
@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with comprehensive status"""
    try:
        # Test database connectivity
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
    """Store content in memory under a specific project with optional image/link support"""
    try:
        memory_id = db.store_memory(
            project=data.project, 
            content=data.content, 
            metadata=data.metadata, 
            tags=data.tags,
            image_url=data.image_url,
            link_url=data.link_url
        )
        
        result_text = f"✅ Memory stored successfully!\n"
        result_text += f"ID: {memory_id}\n"
        result_text += f"Project: {data.project}\n"
        result_text += f"Content: {data.content[:100]}{'...' if len(data.content) > 100 else ''}\n"
        
        if data.tags:
            result_text += f"Tags: {', '.join(data.tags)}\n"
        
        if data.image_url:
            result_text += f"🖼️ Image: Downloaded and stored\n"
        
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
async def memorize_with_file(data: MemorizeWithFileData) -> Dict[str, Any]:
    """Store content with uploaded file (image, document, etc.)"""
    try:
        memory_id = db.store_memory(
            project=data.project,
            content=data.content,
            metadata=data.metadata,
            tags=data.tags,
            file_data=data.file_data,
            file_name=data.file_name,
            file_type=data.file_type
        )
        
        result_text = f"✅ Memory with file stored successfully!\n"
        result_text += f"ID: {memory_id}\n"
        result_text += f"Project: {data.project}\n"
        result_text += f"Content: {data.content[:100]}{'...' if len(data.content) > 100 else ''}\n"
        result_text += f"📎 File: {data.file_name} ({data.file_type})\n"
        
        if data.tags:
            result_text += f"Tags: {', '.join(data.tags)}\n"
        
        return {
            "success": True,
            "memory_id": memory_id,
            "project": data.project,
            "file_name": data.file_name,
            "file_type": data.file_type,
            "message": result_text
        }
    except Exception as e:
        logger.error(f"❌ Error in memorize_with_file: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to store memory with file: {str(e)[:200]}"
        }

@mcp.tool()
async def recall(data: RecallData) -> Dict[str, Any]:
    """Recall memories from a project or search across all memories with file support"""
    try:
        memories = db.recall_memories(data.project, data.query, data.limit, data.include_files)
        
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
                "timestamp": memory['created_at'],
                "has_file": memory['has_file']
            }
            
            if memory.get('tags'):
                formatted_memory['tags'] = memory['tags']
            
            if memory.get('metadata'):
                formatted_memory['metadata'] = memory['metadata']
            
            # Add file info
            if memory.get('file_name'):
                formatted_memory['file_info'] = {
                    "name": memory['file_name'],
                    "type": memory['file_type'],
                    "size": memory['file_size']
                }
                if data.include_files and memory.get('file_data_base64'):
                    formatted_memory['file_data'] = memory['file_data_base64']
            
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
            
            if memory.get('file_info'):
                file_info = memory['file_info']
                size_kb = file_info['size'] / 1024 if file_info['size'] else 0
                result_text += f"📎 File: {file_info['name']} ({file_info['type']}, {size_kb:.1f}KB)\n"
            
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
        
        result_text = f"📊 Memory Database Statistics\n\n"
        result_text += f"📝 Total Memories: {stats['total_memories']}\n"
        result_text += f"📁 Total Projects: {stats['total_projects']}\n"
        result_text += f"🕒 Recent (7 days): {stats['recent_memories_7d']}\n"
        result_text += f"💾 Database: {stats['database_type']}\n\n"
        
        if top_projects:
            result_text += f"🏆 Top Projects:\n"
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

# Auto-activation and natural language processing
@mcp.tool()
async def auto_enable_memory() -> Dict[str, Any]:
    """Auto-enable memory system - detects activation requests and sets up everything"""
    try:
        # Check if already initialized properly
        stats = db.get_stats()
        projects = db.list_projects()
        
        result_text = f"🧠 Memory MCP Server Auto-Enabled!\n\n"
        result_text += f"✅ Status: Active and Ready\n"
        result_text += f"📝 Memories: {stats['total_memories']}\n" 
        result_text += f"📁 Projects: {stats['total_projects']}\n"
        result_text += f"💾 Database: Supabase (Cloud)\n\n"
        
        result_text += f"🚀 Quick Start:\n"
        result_text += f"• Say: 'Memorize this under project [name]: [content]'\n"
        result_text += f"• Say: 'What do you remember about [project]?'\n"
        result_text += f"• Say: 'List all my projects'\n\n
