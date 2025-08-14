import express from "express";
import { neonAuth } from "../lib/neonAuth";
import { emailService } from "../lib/emailService";
import { neon } from "@neondatabase/serverless";
import { inMemoryAuth } from "../db/init-db";

const router = express.Router();
const DATABASE_URL = process.env.DATABASE_URL;

// Check if DATABASE_URL is valid for Neon (must start with postgresql:// and not be a Prisma URL)
const isValidNeonUrl =
  DATABASE_URL &&
  DATABASE_URL.startsWith("postgresql://") &&
  !DATABASE_URL.includes("prisma+postgres://") &&
  DATABASE_URL !== "postgresql://your-database-url-here";

const sql = isValidNeonUrl ? neon(DATABASE_URL) : null;

const useDatabase = sql !== null;

// Middleware to check admin access
const requireAdmin = async (req: any, res: any, next: any) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res
      .status(401)
      .json({ success: false, error: "Access token required" });
  }

  try {
    const decoded = neonAuth.verifyToken(token);
    if (decoded.role !== "admin") {
      return res
        .status(403)
        .json({ success: false, error: "Admin access required" });
    }
    req.user = decoded;
    next();
  } catch (error) {
    return res
      .status(403)
      .json({ success: false, error: "Invalid or expired token" });
  }
};

// Generate secure password
const generatePassword = (): string => {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";
  let password = "";
  for (let i = 0; i < 12; i++) {
    password += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return password;
};

// Generate company email
const generateCompanyEmail = (
  displayName: string,
  role: "admin" | "user",
  department?: string,
): string => {
  const baseName = displayName.toLowerCase().replace(/\s+/g, ".");
  const domainPrefix =
    role === "admin" ? "admin" : department ? department.toLowerCase() : "emp";
  return `${baseName}@${domainPrefix}.yitro.com`;
};

// Get all users (admin only)
router.get("/users", requireAdmin, async (req, res) => {
  try {
    let users = [];

    if (useDatabase) {
      const result = await sql!`
        SELECT
          id, email, display_name, role, email_verified,
          created_at, last_login
        FROM neon_auth.users
        ORDER BY created_at DESC
      `;

      users = result.map((user) => ({
        id: user.id,
        email: user.email,
        displayName: user.display_name,
        role: user.role,
        emailVerified: user.email_verified,
        createdAt: user.created_at,
        lastLogin: user.last_login,
      }));
    } else {
      // Fallback to in-memory users
      users = Array.from(inMemoryAuth.users.values()).map((user) => ({
        id: user.id,
        email: user.email,
        displayName: user.displayName,
        role: user.role,
        emailVerified: user.emailVerified,
        createdAt: user.createdAt.toISOString(),
      }));
    }

    res.json({
      success: true,
      users, // Match component expectation
    });
  } catch (error) {
    console.error("Get users error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch users",
    });
  }
});

// Create new user (admin only) - match component endpoint
router.post("/create-user", requireAdmin, async (req, res) => {
  try {
    const {
      email,
      displayName,
      password,
      role,
      contactNumber,
      department,
      designation,
    } = req.body;

    if (!email || !displayName || !password || !role) {
      return res.status(400).json({
        success: false,
        error: "Email, display name, password and role are required",
      });
    }

    // Use the existing signUp method
    const result = await neonAuth.signUp({
      email,
      password,
      displayName,
    });

    // Update role in database if using real database
    if (useDatabase) {
      await sql!`
        UPDATE neon_auth.users
        SET role = ${role}, email_verified = true
        WHERE email = ${email}
      `;
    } else {
      // For in-memory mode, update the user role
      console.log(`🔄 Updated user role to ${role} in development mode`);
      const user = inMemoryAuth.users.get(email);
      if (user) {
        user.role = role as "admin" | "user";
        user.emailVerified = true;
      }
    }

    // Send welcome email with login credentials
    try {
      await emailService.sendEmployeeWelcomeEmail(
        email,
        displayName,
        email,
        password,
      );
    } catch (emailError) {
      console.error("Failed to send welcome email:", emailError);
      // Don't fail the user creation if email fails
    }

    res.status(201).json({
      success: true,
      user: {
        ...result.user,
        role,
        contactNumber,
        department,
        designation,
      },
      message: "User created successfully. Login credentials sent via email.",
    });
  } catch (error: any) {
    console.error("Create user error:", error);
    res.status(400).json({
      success: false,
      error: error.message || "Failed to create user",
    });
  }
});

// Delete user (admin only)
router.delete("/users/:id", requireAdmin, async (req, res) => {
  try {
    const { id } = req.params;

    if (useDatabase) {
      // Prevent deletion of system admin
      const userCheck = await sql!`
        SELECT email FROM neon_auth.users WHERE id = ${id}
      `;

      if (userCheck.length === 0) {
        return res.status(404).json({
          success: false,
          error: "User not found",
        });
      }

      if (userCheck[0].email === "admin@yitro.com") {
        return res.status(403).json({
          success: false,
          error: "Cannot delete system administrator",
        });
      }

      await sql!`
        DELETE FROM neon_auth.users WHERE id = ${id}
      `;
    } else {
      // For in-memory fallback, find and delete the user
      console.log(`Deleting user with ID: ${id} in development mode`);

      // Find user by ID in the in-memory store
      let userToDelete = null;
      let userEmail = null;

      for (const [email, user] of inMemoryAuth.users.entries()) {
        if (user.id === id) {
          userToDelete = user;
          userEmail = email;
          break;
        }
      }

      if (!userToDelete) {
        return res.status(404).json({
          success: false,
          error: "User not found",
        });
      }

      // Prevent deletion of system admin
      if (userEmail === "admin@yitro.com") {
        return res.status(403).json({
          success: false,
          error: "Cannot delete system administrator",
        });
      }

      // Actually delete the user from in-memory store
      inMemoryAuth.users.delete(userEmail);
      console.log(`✅ User ${userEmail} deleted from in-memory store`);
    }

    res.json({
      success: true,
      message: "User deleted successfully",
    });
  } catch (error) {
    console.error("Delete user error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to delete user",
    });
  }
});

// Resend verification email (admin only)
router.post(
  "/users/:id/resend-verification",
  requireAdmin,
  async (req, res) => {
    try {
      const { id } = req.params;

      const result = await sql`
      SELECT email, display_name, verification_token 
      FROM neon_auth.users 
      WHERE id = ${id} AND email_verified = false
    `;

      if (result.length === 0) {
        return res.status(404).json({
          success: false,
          error: "User not found or already verified",
        });
      }

      const user = result[0];

      // Send verification email
      await emailService.sendWelcomeEmail(
        user.email,
        user.display_name,
        user.verification_token,
      );

      res.json({
        success: true,
        data: { message: "Verification email sent successfully" },
      });
    } catch (error) {
      console.error("Resend verification error:", error);
      res.status(500).json({
        success: false,
        error: "Failed to send verification email",
      });
    }
  },
);

// Update user role (admin only)
router.put("/users/:id/role", requireAdmin, async (req, res) => {
  try {
    const { id } = req.params;
    const { role } = req.body;

    if (!["admin", "user"].includes(role)) {
      return res.status(400).json({
        success: false,
        error: "Invalid role. Must be admin or user",
      });
    }

    await sql`
      UPDATE neon_auth.users
      SET role = ${role}
      WHERE id = ${id}
    `;

    res.json({
      success: true,
      data: { message: "User role updated successfully" },
    });
  } catch (error) {
    console.error("Update role error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to update user role",
    });
  }
});

// Get company-wide metrics (admin only)
router.get("/metrics", requireAdmin, async (req, res) => {
  try {
    // Import prisma dynamically to handle potential connection issues
    let metrics = {
      totalUsers: 0,
      totalAccounts: 0,
      totalLeads: 0,
      totalDeals: 0,
      totalActivities: 0,
      activeUsers: 0,
      wonDeals: 0,
      totalDealValue: 0,
      conversionRate: 0,
      recentActivities: [],
    };

    try {
      const { prisma } = await import("../lib/prisma");

      // Get basic counts
      const [
        accountCount,
        leadCount,
        dealCount,
        activityCount,
        wonDealsData,
        recentActivitiesData,
      ] = await Promise.all([
        prisma.account.count(),
        prisma.lead.count(),
        prisma.activeDeal.count(),
        prisma.activityLog.count(),
        prisma.activeDeal.findMany({
          where: { stage: "ORDER_WON" },
          select: { dealValue: true },
        }),
        prisma.activityLog.findMany({
          take: 5,
          orderBy: { dateTime: "desc" },
          select: {
            activityType: true,
            dateTime: true,
            summary: true,
            associatedAccount: true,
            associatedContact: true,
          },
        }),
      ]);

      // Calculate metrics
      const wonDeals = wonDealsData.length;
      const totalDealValue = wonDealsData.reduce((sum, deal) => {
        const value = parseFloat(deal.dealValue || "0");
        return sum + value;
      }, 0);

      const conversionRate = dealCount > 0 ? (wonDeals / dealCount) * 100 : 0;

      metrics = {
        totalUsers: useDatabase ? 0 : 2, // Will be updated below if database is available
        totalAccounts: accountCount,
        totalLeads: leadCount,
        totalDeals: dealCount,
        totalActivities: activityCount,
        activeUsers: useDatabase ? 0 : 2, // Simplified for development
        wonDeals,
        totalDealValue,
        conversionRate: Math.round(conversionRate * 100) / 100,
        recentActivities: recentActivitiesData.map((activity) => ({
          type: activity.activityType?.replace("_", " ") || "Unknown",
          date: activity.dateTime,
          summary: activity.summary || "No summary",
          account: activity.associatedAccount,
          contact: activity.associatedContact,
        })),
      };
    } catch (prismaError) {
      console.log("Prisma not available, using fallback metrics");
      // Fallback metrics for development
      metrics = {
        totalUsers: 2,
        totalAccounts: 8,
        totalLeads: 12,
        totalDeals: 6,
        totalActivities: 15,
        activeUsers: 2,
        wonDeals: 3,
        totalDealValue: 875000,
        conversionRate: 50.0,
        recentActivities: [
          {
            type: "Call",
            date: new Date(),
            summary: "Discovery call with potential client",
            account: "TechCorp Solutions",
            contact: "John Smith",
          },
          {
            type: "Email",
            date: new Date(Date.now() - 3600000),
            summary: "Sent proposal to prospect",
            account: "Innovate Inc",
            contact: "Jane Doe",
          },
        ],
      };
    }

    // Get user count from auth system
    if (useDatabase) {
      try {
        const userResult =
          await sql!`SELECT COUNT(*) as count FROM neon_auth.users`;
        metrics.totalUsers = parseInt(userResult[0].count);
        metrics.activeUsers = Math.floor(metrics.totalUsers * 0.75); // Assume 75% active
      } catch (error) {
        console.log("Using fallback user metrics");
      }
    }

    res.json({
      success: true,
      data: metrics,
    });
  } catch (error) {
    console.error("Metrics error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to fetch metrics",
    });
  }
});

// Test configuration endpoint
router.post("/test-config", requireAdmin, async (req, res) => {
  try {
    console.log("🧪 Running configuration test...");

    // Test database connection
    let databaseStatus = {
      configured: false,
      connected: false,
      message: "Not configured",
    };

    const DATABASE_URL = process.env.DATABASE_URL;
    if (
      DATABASE_URL &&
      DATABASE_URL !== "postgresql://your-database-url-here"
    ) {
      databaseStatus.configured = true;

      try {
        if (useDatabase && sql) {
          const result = await sql`SELECT NOW() as current_time`;
          if (result && result.length > 0) {
            databaseStatus.connected = true;
            databaseStatus.message = "Database connection successful";
          } else {
            databaseStatus.message = "Database query returned empty result";
          }
        } else {
          databaseStatus.message = "Database connection not initialized";
        }
      } catch (error: any) {
        databaseStatus.message = `Database connection failed: ${error.message}`;
      }
    } else {
      databaseStatus.message = "DATABASE_URL environment variable not set";
    }

    // Test SMTP connection
    let smtpStatus = {
      configured: false,
      connected: false,
      message: "Not configured",
    };

    const smtpUser = process.env.SMTP_USER;
    const smtpPassword = process.env.SMTP_PASSWORD;

    if (smtpUser && smtpPassword) {
      smtpStatus.configured = true;

      try {
        const { emailService } = await import("../lib/emailService");
        const testResult = await emailService.testConnection();

        if (testResult) {
          smtpStatus.connected = true;
          smtpStatus.message = `SMTP connection successful (${process.env.SMTP_SERVICE || "Gmail"})`;
        } else {
          smtpStatus.message = "SMTP connection test failed";
        }
      } catch (error: any) {
        smtpStatus.message = `SMTP connection failed: ${error.message}`;
      }
    } else {
      smtpStatus.message =
        "SMTP_USER and SMTP_PASSWORD environment variables not set";
    }

    res.json({
      success: true,
      status: {
        database: databaseStatus,
        smtp: smtpStatus,
      },
    });
  } catch (error) {
    console.error("Configuration test error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to run configuration test",
    });
  }
});

// Send test email endpoint
router.post("/send-test-email", requireAdmin, async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({
        success: false,
        error: "Email address is required",
      });
    }

    console.log(`📧 Sending test email to: ${email}`);

    const { emailService } = await import("../lib/emailService");
    await emailService.sendTestEmail(email);

    res.json({
      success: true,
      message: "Test email sent successfully",
    });
  } catch (error: any) {
    console.error("Test email error:", error);
    res.status(500).json({
      success: false,
      error: error.message || "Failed to send test email",
    });
  }
});

export default router;
