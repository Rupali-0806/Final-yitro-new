-- Create authentication schema and tables
-- This extends the existing neon_auth schema

-- Add authentication specific tables to neon_auth schema
CREATE TABLE IF NOT EXISTS neon_auth.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    email_verified BOOLEAN DEFAULT false,
    verification_token TEXT,
    password_reset_token TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Add constraints
    CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_role_check CHECK (role IN ('admin', 'user'))
);

-- Create user sessions table for tracking active sessions
CREATE TABLE IF NOT EXISTS neon_auth.user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES neon_auth.users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true
);

-- Create user activity log
CREATE TABLE IF NOT EXISTS neon_auth.user_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES neon_auth.users(id) ON DELETE CASCADE,
    activity_type VARCHAR(100) NOT NULL,
    activity_details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create password history table (optional - for password reuse prevention)
CREATE TABLE IF NOT EXISTS neon_auth.password_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES neon_auth.users(id) ON DELETE CASCADE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON neon_auth.users(email);
CREATE INDEX IF NOT EXISTS idx_users_verification_token ON neon_auth.users(verification_token);
CREATE INDEX IF NOT EXISTS idx_users_password_reset_token ON neon_auth.users(password_reset_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON neon_auth.user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON neon_auth.user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_activity_log_user_id ON neon_auth.user_activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_user_activity_log_created_at ON neon_auth.user_activity_log(created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION neon_auth.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON neon_auth.users 
    FOR EACH ROW EXECUTE FUNCTION neon_auth.update_updated_at_column();

-- Insert default admin user (optional)
INSERT INTO neon_auth.users (email, display_name, password_hash, role, email_verified)
VALUES (
    'admin@yitro.com',
    'Admin User',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewEyFQg3/l/VE6x2', -- password: admin123
    'admin',
    true
) ON CONFLICT (email) DO NOTHING;

-- Create RLS (Row Level Security) policies
ALTER TABLE neon_auth.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE neon_auth.user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE neon_auth.user_activity_log ENABLE ROW LEVEL SECURITY;

-- RLS policy: Users can only see their own data
CREATE POLICY "Users can view own data" ON neon_auth.users
    FOR SELECT USING (id = current_setting('user.id', true)::UUID);

CREATE POLICY "Users can update own data" ON neon_auth.users
    FOR UPDATE USING (id = current_setting('user.id', true)::UUID);

-- RLS policy: Users can only see their own sessions
CREATE POLICY "Users can view own sessions" ON neon_auth.user_sessions
    FOR SELECT USING (user_id = current_setting('user.id', true)::UUID);

-- RLS policy: Users can only see their own activity
CREATE POLICY "Users can view own activity" ON neon_auth.user_activity_log
    FOR SELECT USING (user_id = current_setting('user.id', true)::UUID);

-- Grant necessary permissions
GRANT USAGE ON SCHEMA neon_auth TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON neon_auth.users TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON neon_auth.user_sessions TO PUBLIC;
GRANT SELECT, INSERT ON neon_auth.user_activity_log TO PUBLIC;
GRANT SELECT, INSERT ON neon_auth.password_history TO PUBLIC;
