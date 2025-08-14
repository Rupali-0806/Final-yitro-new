#!/usr/bin/env node

/**
 * Production Configuration Checker
 * Run this to verify your production environment is configured correctly
 */

console.log("🔍 Checking Yitro CRM Production Configuration...\n");

// Check required environment variables
const requiredVars = [
  "DATABASE_URL",
  "STACK_SECRET_SERVER_KEY",
  "NODE_ENV",
  "FRONTEND_URL",
];

const optionalVars = ["SMTP_SERVICE", "SMTP_USER", "SMTP_PASSWORD"];

let hasIssues = false;

console.log("📋 Required Environment Variables:");
requiredVars.forEach((varName) => {
  const value = process.env[varName];
  if (value) {
    if (varName === "DATABASE_URL") {
      if (
        value.includes("postgresql://") &&
        !value.includes("your-database-url-here")
      ) {
        console.log(`✅ ${varName}: Configured (Neon database)`);
      } else {
        console.log(`❌ ${varName}: Invalid - still using placeholder`);
        hasIssues = true;
      }
    } else if (varName === "STACK_SECRET_SERVER_KEY") {
      if (value.length >= 32) {
        console.log(`✅ ${varName}: Configured (${value.length} characters)`);
      } else {
        console.log(`❌ ${varName}: Too short - should be 64+ characters`);
        hasIssues = true;
      }
    } else {
      console.log(`✅ ${varName}: ${value}`);
    }
  } else {
    console.log(`❌ ${varName}: Not set`);
    hasIssues = true;
  }
});

console.log("\n📧 Optional Email Configuration:");
optionalVars.forEach((varName) => {
  const value = process.env[varName];
  if (value) {
    console.log(`✅ ${varName}: Configured`);
  } else {
    console.log(`⚠️  ${varName}: Not set (email features disabled)`);
  }
});

console.log("\n🔧 System Checks:");

// Check Node.js version
const nodeVersion = process.version;
const majorVersion = parseInt(nodeVersion.slice(1).split(".")[0]);
if (majorVersion >= 18) {
  console.log(`✅ Node.js Version: ${nodeVersion} (compatible)`);
} else {
  console.log(`❌ Node.js Version: ${nodeVersion} (requires 18+)`);
  hasIssues = true;
}

// Check if we're in production mode
if (process.env.NODE_ENV === "production") {
  console.log(`✅ Environment: Production mode`);
} else {
  console.log(
    `⚠️  Environment: ${process.env.NODE_ENV || "development"} (should be 'production')`,
  );
}

// Final assessment
console.log("\n" + "=".repeat(50));
if (hasIssues) {
  console.log("❌ Configuration Issues Found!");
  console.log("\n📝 Next Steps:");
  console.log("1. Set missing environment variables in Netlify dashboard");
  console.log("2. Ensure DATABASE_URL points to your Neon database");
  console.log("3. Generate a secure JWT secret (64+ characters)");
  console.log("4. Redeploy your application");
  process.exit(1);
} else {
  console.log("🎉 Production Configuration Complete!");
  console.log("\n✅ Your Yitro CRM is ready for production deployment");
  console.log("\n🚀 Next Steps:");
  console.log("1. Push your changes to trigger deployment");
  console.log("2. Access your production app");
  console.log("3. Login and change default admin password");
  console.log("4. Test all functionality");
}
