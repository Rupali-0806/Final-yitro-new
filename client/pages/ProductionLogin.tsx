import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/components/RealAuthProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  User,
  Lock,
  Mail,
  UserPlus,
  Eye,
  EyeOff,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

export default function ProductionLogin() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error" | "info";
    text: string;
  } | null>(null);
  const [requiresVerification, setRequiresVerification] = useState(false);

  const navigate = useNavigate();
  const { user, signIn, signUp, requestPasswordReset } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (user && user.emailVerified) {
      navigate("/dashboard");
    }
  }, [user, navigate]);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);

    if (!email || !password) {
      setMessage({ type: "error", text: "Please fill in all required fields" });
      return;
    }

    if (isSignUp) {
      if (!displayName) {
        setMessage({ type: "error", text: "Display name is required" });
        return;
      }

      if (password !== confirmPassword) {
        setMessage({ type: "error", text: "Passwords do not match" });
        return;
      }

      if (password.length < 8) {
        setMessage({
          type: "error",
          text: "Password must be at least 8 characters long",
        });
        return;
      }
    }

    setLoading(true);

    try {
      let result;
      if (isSignUp) {
        result = await signUp(email, password, displayName);
      } else {
        result = await signIn(email, password);
      }

      if (result.success) {
        setMessage({ type: "success", text: result.message || "Success!" });

        if (result.requiresVerification) {
          setRequiresVerification(true);
        } else if (!isSignUp) {
          // Successful login, redirect
          navigate("/dashboard");
        }
      } else {
        setMessage({
          type: "error",
          text: result.message || "Authentication failed",
        });
        if (result.requiresVerification) {
          setRequiresVerification(true);
        }
      }
    } catch (error) {
      setMessage({ type: "error", text: "An unexpected error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);

    if (!email) {
      setMessage({ type: "error", text: "Please enter your email address" });
      return;
    }

    setLoading(true);

    try {
      const result = await requestPasswordReset(email);
      if (result.success) {
        setMessage({
          type: "success",
          text: result.message || "Password reset email sent!",
        });
        setShowForgotPassword(false);
      } else {
        setMessage({
          type: "error",
          text: result.message || "Failed to send reset email",
        });
      }
    } catch (error) {
      setMessage({ type: "error", text: "An unexpected error occurred" });
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemo = (userType: "admin" | "user") => {
    const demoEmail =
      userType === "admin" ? "admin@yitro.com" : "user@yitro.com";
    setEmail(demoEmail);
    setPassword("demo123");
    setIsSignUp(false);
    setShowForgotPassword(false);
  };

  if (requiresVerification) {
    return (
      <div className="min-h-screen bg-login-gradient flex items-center justify-center p-4">
        <div className="w-full max-w-md space-y-8">
          <div className="text-center">
            <img
              src="https://cdn.builder.io/api/v1/image/assets%2F087df647f1e8465b80d17ed1202a1a86%2F874635e44e4546cc93b707faf6deafea?format=webp&width=800"
              alt="Yitro Logo"
              className="mx-auto h-16 w-auto mb-8"
            />
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
            <Mail className="mx-auto h-16 w-16 text-blue-500 mb-4" />
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
              Check Your Email! 📧
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              We've sent a verification email to <strong>{email}</strong>
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
              Click the link in the email to verify your account and complete
              the setup.
            </p>
            <div className="space-y-3">
              <Button
                onClick={() => setRequiresVerification(false)}
                variant="outline"
                className="w-full"
              >
                Back to Login
              </Button>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Didn't receive the email? Check your spam folder.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-login-gradient flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        {/* Logo */}
        <div className="text-center">
          <img
            src="https://cdn.builder.io/api/v1/image/assets%2F087df647f1e8465b80d17ed1202a1a86%2F874635e44e4546cc93b707faf6deafea?format=webp&width=800"
            alt="Yitro Logo"
            className="mx-auto h-16 w-auto mb-8"
          />
          <div className="inline-flex items-center justify-center w-24 h-24 border-2 border-white/30 rounded-full bg-white/10 backdrop-blur-sm">
            <User className="w-12 h-12 text-white" strokeWidth={1.5} />
          </div>
        </div>

        {/* Quick Demo Access */}
        {!showForgotPassword && (
          <div className="space-y-3">
            <Button
              onClick={() => handleQuickDemo("admin")}
              className="w-full h-12 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-semibold uppercase tracking-wider shadow-lg transition-all duration-200"
            >
              🔑 Admin Demo
            </Button>

            <Button
              onClick={() => handleQuickDemo("user")}
              className="w-full h-12 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-semibold uppercase tracking-wider shadow-lg transition-all duration-200"
            >
              👤 User Demo
            </Button>
          </div>
        )}

        {/* Authentication Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
          {/* Header */}
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {showForgotPassword
                ? "Reset Password"
                : isSignUp
                  ? "Create Account"
                  : "Welcome Back"}
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              {showForgotPassword
                ? "Enter your email to receive a password reset link"
                : isSignUp
                  ? "Join Yitro CRM platform today"
                  : "Sign in to your Yitro CRM account"}
            </p>
          </div>

          {/* Message Display */}
          {message && (
            <div
              className={`mb-4 p-3 rounded-lg border ${
                message.type === "success"
                  ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
                  : message.type === "error"
                    ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800"
                    : "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800"
              }`}
            >
              <div className="flex items-center space-x-2">
                {message.type === "success" ? (
                  <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                )}
                <p
                  className={`text-sm ${
                    message.type === "success"
                      ? "text-green-600 dark:text-green-400"
                      : message.type === "error"
                        ? "text-red-600 dark:text-red-400"
                        : "text-blue-600 dark:text-blue-400"
                  }`}
                >
                  {message.text}
                </p>
              </div>
            </div>
          )}

          {/* Form */}
          <form
            onSubmit={showForgotPassword ? handleForgotPassword : handleAuth}
            className="space-y-4"
          >
            {/* Display Name (Sign Up only) */}
            {isSignUp && !showForgotPassword && (
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <UserPlus className="h-5 w-5 text-gray-400" />
                </div>
                <Input
                  type="text"
                  placeholder="Display Name"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="pl-10"
                  required={isSignUp}
                />
              </div>
            )}

            {/* Email */}
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Mail className="h-5 w-5 text-gray-400" />
              </div>
              <Input
                type="email"
                placeholder="Email Address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="pl-10"
                required
              />
            </div>

            {/* Password (not for forgot password) */}
            {!showForgotPassword && (
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 pr-12"
                  required
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5 text-gray-400" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>
            )}

            {/* Confirm Password (Sign Up only) */}
            {isSignUp && !showForgotPassword && (
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <Input
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="Confirm Password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="pl-10 pr-12"
                  required={isSignUp}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-5 w-5 text-gray-400" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>
            )}

            {/* Submit Button */}
            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white"
            >
              {loading
                ? "Please wait..."
                : showForgotPassword
                  ? "Send Reset Email"
                  : isSignUp
                    ? "Create Account"
                    : "Sign In"}
            </Button>
          </form>

          {/* Footer Links */}
          <div className="mt-6 space-y-3 text-center">
            {!showForgotPassword && (
              <>
                <button
                  onClick={() => setShowForgotPassword(true)}
                  className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                >
                  Forgot your password?
                </button>

                <div className="flex items-center justify-center space-x-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {isSignUp
                      ? "Already have an account?"
                      : "Don't have an account?"}
                  </span>
                  <button
                    onClick={() => {
                      setIsSignUp(!isSignUp);
                      setMessage(null);
                    }}
                    className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium"
                  >
                    {isSignUp ? "Sign In" : "Sign Up"}
                  </button>
                </div>
              </>
            )}

            {showForgotPassword && (
              <button
                onClick={() => setShowForgotPassword(false)}
                className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
              >
                Back to Sign In
              </button>
            )}
          </div>
        </div>

        {/* Security Notice */}
        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 text-center">
          <p className="text-white/90 text-sm">
            🔒 Your data is protected with enterprise-grade security
          </p>
        </div>
      </div>
    </div>
  );
}
