import { neon } from "@neondatabase/serverless";
import fs from "fs";
import path from "path";

// Initialize database with schema
export async function initializeDatabase() {
  try {
    const databaseUrl = process.env.DATABASE_URL;

    // Check if DATABASE_URL is valid for Neon (must start with postgresql:// and not be a Prisma URL)
    const isValidNeonUrl =
      databaseUrl &&
      databaseUrl.startsWith("postgresql://") &&
      !databaseUrl.includes("prisma+postgres://") &&
      databaseUrl !== "postgresql://your-database-url-here";

    if (!isValidNeonUrl) {
      console.log(
        "⚠️  DATABASE_URL not configured. Using in-memory authentication (for development only)",
      );
      return false;
    }

    const sql = neon(databaseUrl);

    // Check if neon_auth schema exists
    const schemaCheck = await sql`
      SELECT schema_name 
      FROM information_schema.schemata 
      WHERE schema_name = 'neon_auth'
    `;

    if (schemaCheck.length === 0) {
      console.log("🚀 Creating neon_auth schema...");
      await sql`CREATE SCHEMA IF NOT EXISTS neon_auth`;
    }

    // Check if users table exists
    const tableCheck = await sql`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'neon_auth' AND table_name = 'users'
    `;

    if (tableCheck.length === 0) {
      console.log("🚀 Initializing authentication database...");

      // Create tables directly with proper neon syntax

      // Create users table
      await sql`
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

          CONSTRAINT users_email_check CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'),
          CONSTRAINT users_role_check CHECK (role IN ('admin', 'user'))
        )
      `;

      // Create user sessions table
      await sql`
        CREATE TABLE IF NOT EXISTS neon_auth.user_sessions (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id UUID NOT NULL REFERENCES neon_auth.users(id) ON DELETE CASCADE,
          token_hash TEXT NOT NULL,
          expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          ip_address INET,
          user_agent TEXT,
          is_active BOOLEAN DEFAULT true
        )
      `;

      // Create user activity log
      await sql`
        CREATE TABLE IF NOT EXISTS neon_auth.user_activity_log (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id UUID NOT NULL REFERENCES neon_auth.users(id) ON DELETE CASCADE,
          activity_type VARCHAR(100) NOT NULL,
          activity_details JSONB,
          ip_address INET,
          user_agent TEXT,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
      `;

      // Create password history table
      await sql`
        CREATE TABLE IF NOT EXISTS neon_auth.password_history (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id UUID NOT NULL REFERENCES neon_auth.users(id) ON DELETE CASCADE,
          password_hash TEXT NOT NULL,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
      `;

      // Create indexes
      await sql`CREATE INDEX IF NOT EXISTS idx_users_email ON neon_auth.users(email)`;
      await sql`CREATE INDEX IF NOT EXISTS idx_users_verification_token ON neon_auth.users(verification_token)`;
      await sql`CREATE INDEX IF NOT EXISTS idx_users_password_reset_token ON neon_auth.users(password_reset_token)`;
      await sql`CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON neon_auth.user_sessions(user_id)`;
      await sql`CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON neon_auth.user_sessions(expires_at)`;
      await sql`CREATE INDEX IF NOT EXISTS idx_user_activity_log_user_id ON neon_auth.user_activity_log(user_id)`;
      await sql`CREATE INDEX IF NOT EXISTS idx_user_activity_log_created_at ON neon_auth.user_activity_log(created_at)`;

      // Create function to update updated_at timestamp
      await sql`
        CREATE OR REPLACE FUNCTION neon_auth.update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
      `;

      // Create trigger for updated_at
      await sql`
        CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON neon_auth.users
        FOR EACH ROW EXECUTE FUNCTION neon_auth.update_updated_at_column()
      `;

      // Insert default admin user
      await sql`
        INSERT INTO neon_auth.users (email, display_name, password_hash, role, email_verified)
        VALUES (
          'admin@yitro.com',
          'Admin User',
          '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewEyFQg3/l/VE6x2',
          'admin',
          true
        ) ON CONFLICT (email) DO NOTHING
      `;

      console.log("✅ Database schema initialized successfully");

      // Verify admin user exists
      const adminCheck = await sql`
        SELECT id FROM neon_auth.users WHERE email = 'admin@yitro.com'
      `;

      if (adminCheck.length > 0) {
        console.log("✅ Default admin user is ready");
        console.log("📧 Login: admin@yitro.com");
        console.log("🔑 Password: admin123");
      } else {
        console.log("⚠️  Admin user not created. Please check the schema.");
      }
    } else {
      console.log("✅ Database already initialized");
    }

    return true;
  } catch (error) {
    console.error("❌ Database initialization failed:", error);
    return false;
  }
}

// In-memory fallback for development
export class InMemoryAuthService {
  public users = new Map([
    [
      "admin@yitro.com",
      {
        id: "1",
        email: "admin@yitro.com",
        displayName: "Admin User",
        password: "admin123", // In real app, this would be hashed
        role: "admin" as const,
        emailVerified: true,
        createdAt: new Date(),
      },
    ],
    [
      "user@yitro.com",
      {
        id: "2",
        email: "user@yitro.com",
        displayName: "Test User",
        password: "user123", // In real app, this would be hashed
        role: "user" as const,
        emailVerified: true,
        createdAt: new Date(),
      },
    ],
  ]);

  async signIn(email: string, password: string) {
    const user = this.users.get(email);
    if (!user || user.password !== password) {
      throw new Error("Invalid email or password");
    }

    const { password: _, ...userWithoutPassword } = user;
    return {
      user: userWithoutPassword,
      token: `dev-token-${user.id}`,
    };
  }

  async getUserById(userId: string) {
    for (const user of this.users.values()) {
      if (user.id === userId) {
        const { password: _, ...userWithoutPassword } = user;
        return userWithoutPassword;
      }
    }
    return null;
  }
}

export const inMemoryAuth = new InMemoryAuthService();
