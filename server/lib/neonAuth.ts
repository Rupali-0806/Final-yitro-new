import jwt from "jsonwebtoken";
import bcrypt from "bcrypt";
import { neon } from "@neondatabase/serverless";
import { inMemoryAuth } from "../db/init-db.js";

const DATABASE_URL = process.env.DATABASE_URL;

// Check if DATABASE_URL is valid for Neon (must start with postgresql:// and not be a Prisma URL)
const isValidNeonUrl =
  DATABASE_URL &&
  DATABASE_URL.startsWith("postgresql://") &&
  !DATABASE_URL.includes("prisma+postgres://") &&
  DATABASE_URL !== "postgresql://your-database-url-here";

const sql = isValidNeonUrl ? neon(DATABASE_URL) : null;

const JWT_SECRET = process.env.STACK_SECRET_SERVER_KEY || "fallback-secret";

// Check if we're using real database or fallback
const useDatabase = sql !== null;

export interface User {
  id: string;
  email: string;
  displayName: string;
  role: "admin" | "user";
  emailVerified: boolean;
  createdAt: Date;
  lastLogin?: Date;
}

export interface SignUpRequest {
  email: string;
  password: string;
  displayName: string;
}

export interface SignInRequest {
  email: string;
  password: string;
}

export class NeonAuthService {
  // Hash password
  private async hashPassword(password: string): Promise<string> {
    return await bcrypt.hash(password, 12);
  }

  // Verify password
  private async verifyPassword(
    password: string,
    hashedPassword: string,
  ): Promise<boolean> {
    return await bcrypt.compare(password, hashedPassword);
  }

  // Generate JWT token
  private generateToken(user: User): string {
    return jwt.sign(
      {
        userId: user.id,
        email: user.email,
        role: user.role,
        emailVerified: user.emailVerified,
      },
      JWT_SECRET,
      { expiresIn: "7d" },
    );
  }

  // Verify JWT token
  public verifyToken(token: string): any {
    try {
      return jwt.verify(token, JWT_SECRET);
    } catch (error) {
      throw new Error("Invalid token");
    }
  }

  // Determine user role based on email domain and patterns
  private determineUserRole(email: string): "admin" | "user" {
    // Admin patterns
    if (
      email === "admin@yitro.com" ||
      email.includes("@admin.yitro.com") ||
      email.startsWith("admin@") ||
      email.includes(".admin@")
    ) {
      return "admin";
    }

    // All other company emails are users
    return "user";
  }

  // Create user account
  public async signUp(
    data: SignUpRequest,
  ): Promise<{ user: User; token: string; verificationToken: string }> {
    try {
      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        console.log("🔄 Using in-memory user creation (development mode)");

        // Check if user already exists in memory
        for (const existingUser of inMemoryAuth.users.values()) {
          if (existingUser.email === data.email) {
            throw new Error("User already exists with this email");
          }
        }

        // Create new user in memory
        const newId = (inMemoryAuth.users.size + 1).toString();
        const role = this.determineUserRole(data.email);

        const newUser = {
          id: newId,
          email: data.email,
          displayName: data.displayName,
          password: data.password, // In real app, this would be hashed
          role: role,
          emailVerified: true,
          createdAt: new Date(),
        };

        // Add to in-memory store
        inMemoryAuth.users.set(data.email, newUser);

        const user: User = {
          id: newUser.id,
          email: newUser.email,
          displayName: newUser.displayName,
          role: newUser.role,
          emailVerified: newUser.emailVerified,
          createdAt: newUser.createdAt,
        };

        const token = this.generateToken(user);
        const verificationToken = "dev-verification-token";

        return { user, token, verificationToken };
      }

      // Check if user already exists
      const existingUser = await sql!`
        SELECT id FROM neon_auth.users WHERE email = ${data.email}
      `;

      if (existingUser.length > 0) {
        throw new Error("User already exists with this email");
      }

      // Hash password
      const hashedPassword = await this.hashPassword(data.password);

      // Determine role
      const role = this.determineUserRole(data.email);

      // Generate verification token
      const verificationToken = jwt.sign(
        { email: data.email, action: "verify-email" },
        JWT_SECRET,
        { expiresIn: "24h" },
      );

      // Create user in database
      const result = await sql!`
        INSERT INTO neon_auth.users (
          email,
          display_name,
          password_hash,
          role,
          email_verified,
          verification_token,
          created_at
        )
        VALUES (
          ${data.email},
          ${data.displayName},
          ${hashedPassword},
          ${role},
          false,
          ${verificationToken},
          NOW()
        )
        RETURNING id, email, display_name, role, email_verified, created_at
      `;

      const user: User = {
        id: result[0].id,
        email: result[0].email,
        displayName: result[0].display_name,
        role: result[0].role,
        emailVerified: result[0].email_verified,
        createdAt: result[0].created_at,
      };

      const token = this.generateToken(user);

      return { user, token, verificationToken };
    } catch (error) {
      console.error("SignUp error:", error);
      throw error;
    }
  }

  // Sign in user
  public async signIn(
    data: SignInRequest,
  ): Promise<{ user: User; token: string }> {
    try {
      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        console.log("🔄 Using in-memory authentication (development mode)");
        const result = await inMemoryAuth.signIn(data.email, data.password);
        const token = this.generateToken(result.user);
        return { user: result.user, token };
      }

      // Get user from database
      const result = await sql!`
        SELECT
          id, email, display_name, password_hash, role,
          email_verified, created_at, last_login
        FROM neon_auth.users
        WHERE email = ${data.email}
      `;

      if (result.length === 0) {
        throw new Error("Invalid email or password");
      }

      const userData = result[0];

      // Verify password
      const isValidPassword = await this.verifyPassword(
        data.password,
        userData.password_hash,
      );
      if (!isValidPassword) {
        throw new Error("Invalid email or password");
      }

      // Check if email is verified
      if (!userData.email_verified) {
        throw new Error("Please verify your email before signing in");
      }

      // Update last login
      await sql!`
        UPDATE neon_auth.users
        SET last_login = NOW()
        WHERE id = ${userData.id}
      `;

      const user: User = {
        id: userData.id,
        email: userData.email,
        displayName: userData.display_name,
        role: userData.role,
        emailVerified: userData.email_verified,
        createdAt: userData.created_at,
        lastLogin: new Date(),
      };

      const token = this.generateToken(user);

      return { user, token };
    } catch (error) {
      console.error("SignIn error:", error);
      throw error;
    }
  }

  // Verify email
  public async verifyEmail(verificationToken: string): Promise<User> {
    try {
      // Verify token
      const decoded = jwt.verify(verificationToken, JWT_SECRET) as any;

      if (decoded.action !== "verify-email") {
        throw new Error("Invalid verification token");
      }

      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        console.log("🔄 Using in-memory email verification (development mode)");

        const user = inMemoryAuth.users.get(decoded.email);
        if (!user) {
          throw new Error("Invalid or expired verification token");
        }

        // Update verification status in memory
        user.emailVerified = true;
        inMemoryAuth.users.set(decoded.email, user);

        const { password: _, ...userWithoutPassword } = user;
        return userWithoutPassword;
      }

      // Update user email verification status
      const result = await sql!`
        UPDATE neon_auth.users
        SET email_verified = true, verification_token = null
        WHERE email = ${decoded.email} AND verification_token = ${verificationToken}
        RETURNING id, email, display_name, role, email_verified, created_at
      `;

      if (result.length === 0) {
        throw new Error("Invalid or expired verification token");
      }

      const userData = result[0];
      return {
        id: userData.id,
        email: userData.email,
        displayName: userData.display_name,
        role: userData.role,
        emailVerified: userData.email_verified,
        createdAt: userData.created_at,
      };
    } catch (error) {
      console.error("Email verification error:", error);
      throw error;
    }
  }

  // Request password reset
  public async requestPasswordReset(email: string): Promise<string> {
    try {
      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        console.log(
          "🔄 Password reset in development mode - returning mock token",
        );
        return jwt.sign({ email, action: "reset-password" }, JWT_SECRET, {
          expiresIn: "1h",
        });
      }

      // Check if user exists
      const result = await sql!`
        SELECT id FROM neon_auth.users WHERE email = ${email}
      `;

      if (result.length === 0) {
        throw new Error("No account found with this email");
      }

      // Generate reset token
      const resetToken = jwt.sign(
        { email, action: "reset-password" },
        JWT_SECRET,
        { expiresIn: "1h" },
      );

      // Store reset token
      await sql!`
        UPDATE neon_auth.users
        SET password_reset_token = ${resetToken}
        WHERE email = ${email}
      `;

      return resetToken;
    } catch (error) {
      console.error("Password reset request error:", error);
      throw error;
    }
  }

  // Reset password
  public async resetPassword(
    resetToken: string,
    newPassword: string,
  ): Promise<User> {
    try {
      // Verify token
      const decoded = jwt.verify(resetToken, JWT_SECRET) as any;

      if (decoded.action !== "reset-password") {
        throw new Error("Invalid reset token");
      }

      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        console.log("🔄 Using in-memory password reset (development mode)");

        const user = inMemoryAuth.users.get(decoded.email);
        if (!user) {
          throw new Error("Invalid or expired reset token");
        }

        // Update password in memory
        user.password = newPassword;
        inMemoryAuth.users.set(decoded.email, user);

        const { password: _, ...userWithoutPassword } = user;
        return userWithoutPassword;
      }

      // Hash new password
      const hashedPassword = await this.hashPassword(newPassword);

      // Update password
      const result = await sql!`
        UPDATE neon_auth.users
        SET password_hash = ${hashedPassword}, password_reset_token = null
        WHERE email = ${decoded.email} AND password_reset_token = ${resetToken}
        RETURNING id, email, display_name, role, email_verified, created_at
      `;

      if (result.length === 0) {
        throw new Error("Invalid or expired reset token");
      }

      const userData = result[0];
      return {
        id: userData.id,
        email: userData.email,
        displayName: userData.display_name,
        role: userData.role,
        emailVerified: userData.email_verified,
        createdAt: userData.created_at,
      };
    } catch (error) {
      console.error("Password reset error:", error);
      throw error;
    }
  }

  // Change password (for logged-in users)
  public async changePassword(
    userId: string,
    currentPassword: string,
    newPassword: string,
  ): Promise<void> {
    try {
      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        console.log("🔄 Using in-memory password change (development mode)");

        // Find user in memory by ID
        let foundUser = null;
        for (const user of inMemoryAuth.users.values()) {
          if (user.id === userId) {
            foundUser = user;
            break;
          }
        }

        if (!foundUser) {
          throw new Error("User not found");
        }

        // Verify current password (in dev mode, passwords are stored in plain text)
        if (foundUser.password !== currentPassword) {
          throw new Error("Current password is incorrect");
        }

        // Update password in memory
        foundUser.password = newPassword;
        inMemoryAuth.users.set(foundUser.email, foundUser);

        console.log("✅ Password changed successfully in memory");
        return;
      }

      // Get current password hash
      const result = await sql!`
        SELECT password_hash FROM neon_auth.users WHERE id = ${userId}
      `;

      if (result.length === 0) {
        throw new Error("User not found");
      }

      // Verify current password
      const isValidPassword = await this.verifyPassword(
        currentPassword,
        result[0].password_hash,
      );
      if (!isValidPassword) {
        throw new Error("Current password is incorrect");
      }

      // Hash new password
      const hashedPassword = await this.hashPassword(newPassword);

      // Update password
      await sql!`
        UPDATE neon_auth.users
        SET password_hash = ${hashedPassword}
        WHERE id = ${userId}
      `;
    } catch (error) {
      console.error("Change password error:", error);
      throw error;
    }
  }

  // Get user by ID
  public async getUserById(userId: string): Promise<User | null> {
    try {
      // Use in-memory auth if database is not configured
      if (!useDatabase) {
        return await inMemoryAuth.getUserById(userId);
      }

      const result = await sql!`
        SELECT id, email, display_name, role, email_verified, created_at, last_login
        FROM neon_auth.users
        WHERE id = ${userId}
      `;

      if (result.length === 0) {
        return null;
      }

      const userData = result[0];
      return {
        id: userData.id,
        email: userData.email,
        displayName: userData.display_name,
        role: userData.role,
        emailVerified: userData.email_verified,
        createdAt: userData.created_at,
        lastLogin: userData.last_login,
      };
    } catch (error) {
      console.error("Get user error:", error);
      throw error;
    }
  }
}

export const neonAuth = new NeonAuthService();
