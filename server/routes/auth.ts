import express from "express";
import { neonAuth, User } from "../lib/neonAuth";
import { emailService } from "../lib/emailService";

const router = express.Router();

// Middleware to extract user info from token
const authenticateToken = async (req: any, res: any, next: any) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res
      .status(401)
      .json({ success: false, error: "Access token required" });
  }

  try {
    const decoded = neonAuth.verifyToken(token);
    req.user = decoded;
    next();
  } catch (error) {
    return res
      .status(403)
      .json({ success: false, error: "Invalid or expired token" });
  }
};

// Sign up endpoint
router.post("/signup", async (req, res) => {
  try {
    const { email, password, displayName } = req.body;

    // Validate input
    if (!email || !password || !displayName) {
      return res.status(400).json({
        success: false,
        error: "Email, password, and display name are required",
      });
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        success: false,
        error: "Invalid email format",
      });
    }

    // Validate password strength
    if (password.length < 8) {
      return res.status(400).json({
        success: false,
        error: "Password must be at least 8 characters long",
      });
    }

    const result = await neonAuth.signUp({ email, password, displayName });

    // Send welcome email with verification
    try {
      await emailService.sendWelcomeEmail(
        email,
        displayName,
        result.verificationToken,
      );
    } catch (emailError) {
      console.error("Failed to send welcome email:", emailError);
      // Don't fail the signup if email fails
    }

    res.status(201).json({
      success: true,
      data: {
        user: result.user,
        token: result.token,
        message:
          "Account created successfully. Please check your email to verify your account.",
      },
    });
  } catch (error: any) {
    console.error("Signup error:", error);
    res.status(400).json({
      success: false,
      error: error.message || "Failed to create account",
    });
  }
});

// Sign in endpoint
router.post("/signin", async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        success: false,
        error: "Email and password are required",
      });
    }

    const result = await neonAuth.signIn({ email, password });

    // Send login notification email
    try {
      const loginDetails = {
        ip: req.ip || req.connection.remoteAddress,
        userAgent: req.headers["user-agent"],
        timestamp: new Date(),
      };
      await emailService.sendLoginNotification(
        email,
        result.user.displayName,
        loginDetails,
      );
    } catch (emailError) {
      console.error("Failed to send login notification:", emailError);
      // Don't fail the login if email fails
    }

    res.json({
      success: true,
      data: {
        user: result.user,
        token: result.token,
        message: "Signed in successfully",
      },
    });
  } catch (error: any) {
    console.error("Signin error:", error);
    res.status(401).json({
      success: false,
      error: error.message || "Invalid credentials",
    });
  }
});

// Verify email endpoint
router.post("/verify-email", async (req, res) => {
  try {
    const { token } = req.body;

    if (!token) {
      return res.status(400).json({
        success: false,
        error: "Verification token is required",
      });
    }

    const user = await neonAuth.verifyEmail(token);

    res.json({
      success: true,
      data: {
        user,
        message: "Email verified successfully",
      },
    });
  } catch (error: any) {
    console.error("Email verification error:", error);
    res.status(400).json({
      success: false,
      error: error.message || "Invalid verification token",
    });
  }
});

// Request password reset
router.post("/request-password-reset", async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({
        success: false,
        error: "Email is required",
      });
    }

    const resetToken = await neonAuth.requestPasswordReset(email);

    // Send password reset email
    try {
      await emailService.sendPasswordResetEmail(email, "User", resetToken);
    } catch (emailError) {
      console.error("Failed to send password reset email:", emailError);
      throw new Error("Failed to send password reset email");
    }

    res.json({
      success: true,
      data: {
        message: "Password reset email sent. Please check your inbox.",
      },
    });
  } catch (error: any) {
    console.error("Password reset request error:", error);
    res.status(400).json({
      success: false,
      error: error.message || "Failed to process password reset request",
    });
  }
});

// Reset password
router.post("/reset-password", async (req, res) => {
  try {
    const { token, newPassword } = req.body;

    if (!token || !newPassword) {
      return res.status(400).json({
        success: false,
        error: "Reset token and new password are required",
      });
    }

    // Validate password strength
    if (newPassword.length < 8) {
      return res.status(400).json({
        success: false,
        error: "Password must be at least 8 characters long",
      });
    }

    const user = await neonAuth.resetPassword(token, newPassword);

    res.json({
      success: true,
      data: {
        user,
        message: "Password reset successfully",
      },
    });
  } catch (error: any) {
    console.error("Password reset error:", error);
    res.status(400).json({
      success: false,
      error: error.message || "Failed to reset password",
    });
  }
});

// Change password (for logged-in users)
router.post("/change-password", authenticateToken, async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body;
    const userId = req.user.userId;

    if (!currentPassword || !newPassword) {
      return res.status(400).json({
        success: false,
        error: "Current password and new password are required",
      });
    }

    // Validate password strength
    if (newPassword.length < 8) {
      return res.status(400).json({
        success: false,
        error: "New password must be at least 8 characters long",
      });
    }

    await neonAuth.changePassword(userId, currentPassword, newPassword);

    res.json({
      success: true,
      data: {
        message: "Password changed successfully",
      },
    });
  } catch (error: any) {
    console.error("Change password error:", error);
    res.status(400).json({
      success: false,
      error: error.message || "Failed to change password",
    });
  }
});

// Get current user profile
router.get("/profile", authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;
    const user = await neonAuth.getUserById(userId);

    if (!user) {
      return res.status(404).json({
        success: false,
        error: "User not found",
      });
    }

    res.json({
      success: true,
      data: { user },
    });
  } catch (error: any) {
    console.error("Get profile error:", error);
    res.status(500).json({
      success: false,
      error: "Failed to get user profile",
    });
  }
});

// Validate token endpoint
router.post("/validate-token", async (req, res) => {
  try {
    const { token } = req.body;

    if (!token) {
      return res.status(400).json({
        success: false,
        error: "Token is required",
      });
    }

    const decoded = neonAuth.verifyToken(token);
    const user = await neonAuth.getUserById(decoded.userId);

    if (!user) {
      return res.status(404).json({
        success: false,
        error: "User not found",
      });
    }

    res.json({
      success: true,
      data: { user, valid: true },
    });
  } catch (error: any) {
    res.status(401).json({
      success: false,
      error: "Invalid token",
      valid: false,
    });
  }
});

export default router;
