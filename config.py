"""
Configuration file for evo-ssearch oldapp.py
Contains all configurable settings with environment variable support
"""
import os

class Config:
    # Server configuration
    HOST = os.getenv('EVOSSEARCH_HOST', '0.0.0.0')  # 0.0.0.0 allows network access
    PORT = int(os.getenv('EVOSSEARCH_PORT', '5000'))
    DEBUG = os.getenv('EVOSSEARCH_DEBUG', 'False').lower() in ('true', '1', 'yes', 'on')
    
    # CLIP model configuration
    CLIP_MODEL = os.getenv('EVOSSEARCH_CLIP_MODEL', 'ViT-B/32')
    
    # Search result limits
    MIN_RESULTS = int(os.getenv('EVOSSEARCH_MIN_RESULTS', '3'))
    MAX_RESULTS = int(os.getenv('EVOSSEARCH_MAX_RESULTS', '48'))
    DEFAULT_RESULTS = int(os.getenv('EVOSSEARCH_DEFAULT_RESULTS', '12'))
    
    # Processing configuration
    BATCH_SIZE = int(os.getenv('EVOSSEARCH_BATCH_SIZE', '32'))
    THUMBNAIL_SIZE = (400, 400)
    THUMBNAIL_QUALITY = int(os.getenv('EVOSSEARCH_THUMBNAIL_QUALITY', '85'))
    
    # File system configuration
    INDEX_FOLDER_NAME = os.getenv('EVOSSEARCH_INDEX_FOLDER', '.clip_index')
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # Comment system configuration
    MAX_COMMENT_LENGTH = int(os.getenv('EVOSSEARCH_MAX_COMMENT_LENGTH', '100'))
    
    # Security configuration
    MAX_FILE_SIZE_MB = int(os.getenv('EVOSSEARCH_MAX_FILE_SIZE_MB', '50'))
    
    @classmethod
    def get_server_urls(cls):
        """Get list of server URLs for display"""
        import socket
        
        urls = []
        
        # Always include localhost
        urls.append(f"http://localhost:{cls.PORT}")
        
        # Add network IPs if binding to all interfaces
        if cls.HOST == '0.0.0.0':
            try:
                # Get local IP address
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    urls.append(f"http://{local_ip}:{cls.PORT}")
            except:
                pass
            
            # Try to get all network interfaces
            try:
                hostname = socket.gethostname()
                for addr_info in socket.getaddrinfo(hostname, None):
                    ip = addr_info[4][0]
                    if ip not in ['127.0.0.1', '::1'] and not ip.startswith('169.254'):
                        url = f"http://{ip}:{cls.PORT}"
                        if url not in urls:
                            urls.append(url)
            except:
                pass
        
        return urls
    
    @classmethod
    def print_startup_info(cls):
        """Print configuration info on startup"""
        print("=" * 60)
        print("evo-ssearch (oldapp.py) - CLIP Image Search Server")
        print("=" * 60)
        print(f"Host: {cls.HOST}")
        print(f"Port: {cls.PORT}")
        print(f"Debug: {cls.DEBUG}")
        print(f"CLIP Model: {cls.CLIP_MODEL}")
        print(f"Result Limits: {cls.MIN_RESULTS}-{cls.MAX_RESULTS} (default: {cls.DEFAULT_RESULTS})")
        print()
        print("Server available at:")
        for url in cls.get_server_urls():
            print(f"  {url}")
        print()
        print("Use Ctrl+C to stop the server")
        print("=" * 60)

# Create a default config instance
config = Config()