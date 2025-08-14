import { neon } from "@neondatabase/serverless";
import bcrypt from "bcrypt";

// Setup test accounts for development
export async function setupTestAccounts() {
  try {
    const databaseUrl = process.env.DATABASE_URL;

    // Check if DATABASE_URL is valid for Neon (must start with postgresql:// and not be a Prisma URL)
    const isValidNeonUrl =
      databaseUrl &&
      databaseUrl.startsWith("postgresql://") &&
      !databaseUrl.includes("prisma+postgres://") &&
      databaseUrl !== "postgresql://your-database-url-here";

    if (!isValidNeonUrl) {
      console.log("📝 Setting up in-memory test accounts...");
      return setupInMemoryAccounts();
    }

    const sql = neon(databaseUrl);

    // Hash passwords
    const adminPassword = await bcrypt.hash("admin123", 12);
    const userPassword = await bcrypt.hash("user123", 12);

    console.log("🔐 Creating test accounts...");

    // Delete existing test accounts
    await sql`DELETE FROM neon_auth.users WHERE email IN ('admin@yitro.com', 'user@yitro.com')`;

    // Create admin account
    await sql`
      INSERT INTO neon_auth.users (
        email, display_name, password_hash, role, email_verified, created_at
      ) VALUES (
        'admin@yitro.com',
        'Admin User',
        ${adminPassword},
        'admin',
        true,
        NOW()
      )
    `;

    // Create test user account
    await sql`
      INSERT INTO neon_auth.users (
        email, display_name, password_hash, role, email_verified, created_at
      ) VALUES (
        'user@yitro.com',
        'Test User',
        ${userPassword},
        'user',
        true,
        NOW()
      )
    `;

    console.log("✅ Test accounts created successfully!");
    console.log("🔑 Admin Login - Email: admin@yitro.com | Password: admin123");
    console.log("🔑 User Login - Email: user@yitro.com | Password: user123");

    return true;
  } catch (error) {
    console.error("❌ Failed to setup test accounts:", error);
    return false;
  }
}

function setupInMemoryAccounts() {
  console.log("✅ In-memory test accounts ready!");
  console.log("🔑 Admin Login - Email: admin@yitro.com | Password: admin123");
  console.log("🔑 User Login - Email: user@yitro.com | Password: user123");
  return true;
}
