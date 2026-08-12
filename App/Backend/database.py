from supabase import create_client

try:
    from .config import SUPABASE_URL, SUPABASE_KEY
except ImportError:
    from config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)