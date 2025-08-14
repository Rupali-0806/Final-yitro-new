#!/usr/bin/env tsx

/**
 * Configuration Test Script
 *
 * This script tests both database and SMTP configuration
 * to ensure your production setup is working correctly.
 *
 * Usage:
 *   npm run test-config
 *   or
 *   npx tsx server/scripts/test-config.ts
 */

import { neon } from "@neondatabase/serverless";
import { emailService } from "../lib/emailService";
import * as dotenv from "dotenv";

// Load environment variables
dotenv.config();

interface TestResults {
  database: {
    configured: boolean;
    connected: boolean;
    error?: string;
  };
  smtp: {
    configured: boolean;
    connected: boolean;
    error?: string;
  };
}

async function testDatabaseConnection(): Promise<TestResults["database"]> {
  console.log("\n🗄️  Testing Database Connection...");

  const databaseUrl = process.env.DATABASE_URL;

  // Check if DATABASE_URL is valid for Neon (must start with postgresql:// and not be a Prisma URL)
  const isValidNeonUrl =
    databaseUrl &&
    databaseUrl.startsWith("postgresql://") &&
    !databaseUrl.includes("prisma+postgres://") &&
    databaseUrl !== "postgresql://your-database-url-here";

  if (!isValidNeonUrl) {
    console.log("❌ DATABASE_URL not configured or invalid for Neon");
    return {
      configured: false,
      connected: false,
      error:
        "DATABASE_URL environment variable not set, using placeholder value, or contains Prisma format",
    };
  }

  console.log("✅ DATABASE_URL configured");

  try {
    const sql = neon(databaseUrl);

    // Test basic connection
    const result =
      await sql`SELECT NOW() as current_time, version() as version`;

    if (result && result.length > 0) {
      console.log("✅ Database connection successful!");
      console.log(`   Server time: ${result[0].current_time}`);
      console.log(`   PostgreSQL version: ${result[0].version?.split(" ")[0]}`);

      // Test schema existence
      try {
        const schemaCheck = await sql`
          SELECT schema_name 
          FROM information_schema.schemata 
          WHERE schema_name = 'neon_auth'
        `;

        if (schemaCheck.length > 0) {
          console.log("✅ Authentication schema exists");

          // Check if tables exist
          const tableCheck = await sql`
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'neon_auth'
          `;

          console.log(`✅ Found ${tableCheck.length} authentication tables`);
        } else {
          console.log("⚠️  Authentication schema will be created on first run");
        }
      } catch (schemaError) {
        console.log("ℹ️  Schema check skipped (will be created automatically)");
      }

      return {
        configured: true,
        connected: true,
      };
    } else {
      throw new Error("Empty result from database query");
    }
  } catch (error: any) {
    console.log("❌ Database connection failed:");
    console.log(`   Error: ${error.message}`);

    return {
      configured: true,
      connected: false,
      error: error.message,
    };
  }
}

async function testSMTPConnection(): Promise<TestResults["smtp"]> {
  console.log("\n📧 Testing SMTP Connection...");

  const smtpUser = process.env.SMTP_USER;
  const smtpPassword = process.env.SMTP_PASSWORD;

  if (!smtpUser || !smtpPassword) {
    console.log("❌ SMTP credentials not configured");
    console.log(
      "   Missing: SMTP_USER and/or SMTP_PASSWORD environment variables",
    );
    return {
      configured: false,
      connected: false,
      error: "SMTP credentials not configured",
    };
  }

  console.log("✅ SMTP credentials configured");
  console.log(`   SMTP User: ${smtpUser}`);
  console.log(
    `   SMTP Service: ${process.env.SMTP_SERVICE || "gmail (default)"}`,
  );
  console.log(`   SMTP Host: ${process.env.SMTP_HOST || "gmail service"}`);
  console.log(`   SMTP Port: ${process.env.SMTP_PORT || "587 (default)"}`);

  try {
    const connected = await emailService.testConnection();

    if (connected) {
      console.log("✅ SMTP connection successful!");
      return {
        configured: true,
        connected: true,
      };
    } else {
      return {
        configured: true,
        connected: false,
        error: "SMTP connection test failed",
      };
    }
  } catch (error: any) {
    console.log("❌ SMTP connection failed:");
    console.log(`   Error: ${error.message}`);

    return {
      configured: true,
      connected: false,
      error: error.message,
    };
  }
}

async function sendTestEmail(): Promise<void> {
  const testEmail = process.env.TEST_EMAIL || process.env.SMTP_USER;

  if (!testEmail) {
    console.log(
      "⚠️  No test email provided. Set TEST_EMAIL environment variable to send a test email.",
    );
    return;
  }

  console.log(`\n📨 Sending test email to: ${testEmail}`);

  try {
    await emailService.sendTestEmail(testEmail);
    console.log("✅ Test email sent successfully!");
    console.log("   Check your inbox (and spam folder) for the test email.");
  } catch (error: any) {
    console.log("❌ Failed to send test email:");
    console.log(`   Error: ${error.message}`);
  }
}

function printConfigurationGuide(results: TestResults): void {
  console.log("\n" + "=".repeat(80));
  console.log("📋 CONFIGURATION SUMMARY");
  console.log("=".repeat(80));

  // Database summary
  console.log("\n🗄️  DATABASE:");
  if (results.database.configured && results.database.connected) {
    console.log("   ✅ Status: Ready for production");
  } else if (results.database.configured && !results.database.connected) {
    console.log("   ❌ Status: Configured but connection failed");
    console.log(`   ❌ Error: ${results.database.error}`);
  } else {
    console.log("   ⚠️  Status: Not configured (using in-memory fallback)");
  }

  // SMTP summary
  console.log("\n📧 EMAIL:");
  if (results.smtp.configured && results.smtp.connected) {
    console.log("   ✅ Status: Ready for production");
  } else if (results.smtp.configured && !results.smtp.connected) {
    console.log("   ❌ Status: Configured but connection failed");
    console.log(`   ❌ Error: ${results.smtp.error}`);
  } else {
    console.log("   ⚠️  Status: Not configured (emails will not be sent)");
  }

  // Overall status
  console.log("\n🎯 OVERALL STATUS:");
  const dbReady = results.database.connected;
  const emailReady = results.smtp.connected;

  if (dbReady && emailReady) {
    console.log("   🎉 PRODUCTION READY!");
    console.log(
      "   Your application is fully configured for production deployment.",
    );
  } else if (dbReady || emailReady) {
    console.log("   ⚠️  PARTIALLY CONFIGURED");
    console.log("   Some features may not work as expected in production.");
  } else {
    console.log("   🛠️  DEVELOPMENT MODE");
    console.log("   Application will use fallback configurations.");
  }

  // Configuration help
  console.log("\n📖 NEXT STEPS:");

  if (!results.database.configured) {
    console.log("   📝 Set DATABASE_URL environment variable");
    console.log(
      "      Example: postgresql://user:pass@host:5432/database?sslmode=require",
    );
  }

  if (!results.smtp.configured) {
    console.log("   📝 Set SMTP environment variables:");
    console.log("      SMTP_USER=your-email@gmail.com");
    console.log("      SMTP_PASSWORD=your-app-password");
    console.log("      SMTP_SERVICE=gmail (or sendgrid, mailgun, etc.)");
  }

  if (results.database.configured && !results.database.connected) {
    console.log("   🔧 Fix database connection:");
    console.log("      - Verify connection string format");
    console.log("      - Check network connectivity");
    console.log("      - Ensure database server is running");
    console.log("      - Verify SSL/TLS configuration");
  }

  if (results.smtp.configured && !results.smtp.connected) {
    console.log("   🔧 Fix SMTP connection:");
    console.log("      - Verify username/password are correct");
    console.log(
      "      - For Gmail: use App-specific password, not account password",
    );
    console.log("      - Check SMTP service configuration");
    console.log("      - Ensure 2FA is enabled (for Gmail)");
  }

  console.log("\n📚 For detailed setup instructions, see:");
  console.log("   📖 PRODUCTION_SETUP.md");

  console.log("\n" + "=".repeat(80));
}

async function main(): Promise<void> {
  console.log("🚀 Yitro CRM - Configuration Test");
  console.log("This script will test your database and SMTP configuration.\n");

  try {
    // Test database connection
    const databaseResults = await testDatabaseConnection();

    // Test SMTP connection
    const smtpResults = await testSMTPConnection();

    // Combine results
    const results: TestResults = {
      database: databaseResults,
      smtp: smtpResults,
    };

    // Send test email if SMTP is working
    if (results.smtp.connected) {
      await sendTestEmail();
    }

    // Print configuration summary
    printConfigurationGuide(results);
  } catch (error) {
    console.error("\n❌ Configuration test failed:", error);
    process.exit(1);
  }
}

// Run the configuration test
if (require.main === module) {
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

export { main as testConfiguration };
