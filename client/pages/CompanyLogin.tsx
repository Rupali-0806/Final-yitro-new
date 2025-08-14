import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/components/RealAuthProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  User,
  Lock,
  Mail,
  Eye,
  EyeOff,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

export default function CompanyLogin() {
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{
    type: "success" | "error" | "info";
    text: string;
  } | null>(null);
  const [requiresVerification, setRequiresVerification] = useState(false);

  const navigate = useNavigate();
  const { user, signIn, requestPasswordReset } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (user && user.emailVerified) {
      navigate("/dashboard");
    }
  }, [user, navigate]);

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);

    if (!email || !password) {
      setMessage({
        type: "error",
        text: "Please enter your email and password",
      });
      return;
    }

    setLoading(true);

    try {
      const result = await signIn(email, password);

      if (result.success) {
        setMessage({
          type: "success",
          text: result.message || "Welcome back!",
        });
        navigate("/dashboard");
      } else {
        setMessage({
          type: "error",
          text: result.message || "Invalid email or password",
        });
        if (result.requiresVerification) {
          setRequiresVerification(true);
        }
      }
    } catch (error) {
      setMessage({
        type: "error",
        text: "Connection error. Please try again.",
      });
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
      setMessage({
        type: "error",
        text: "Connection error. Please try again.",
      });
    } finally {
      setLoading(false);
    }
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

          <div className="bg-white/20 backdrop-blur-sm rounded-lg shadow-lg p-8 text-center border border-white/30">
            <Mail className="mx-auto h-16 w-16 text-white mb-4" />
            <h1 className="text-2xl font-bold text-white mb-2">
              Check Your Email! 📧
            </h1>
            <p className="text-white/90 mb-4">
              We've sent a verification email to <strong>{email}</strong>
            </p>
            <p className="text-sm text-white/70 mb-6">
              Click the link in the email to verify your account and complete
              the setup.
            </p>
            <Button
              onClick={() => setRequiresVerification(false)}
              variant="outline"
              className="w-full bg-white/10 border-white/30 text-white hover:bg-white/20"
            >
              Back to Login
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-login-gradient flex items-center justify-center p-4">
      <div className="w-full max-w-sm space-y-8">
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

        {/* Login Form */}
        <form
          onSubmit={showForgotPassword ? handleForgotPassword : handleSignIn}
          className="space-y-6"
        >
          {/* Header */}
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-white mb-2">
              {showForgotPassword ? "Reset Password" : "Welcome Back"}
            </h1>
            <p className="text-white/80">
              {showForgotPassword
                ? "Enter your email to receive a password reset link"
                : "Sign in to your Yitro CRM account"}
            </p>
          </div>

          {/* Message Display */}
          {message && (
            <div
              className={`mb-4 p-3 rounded-lg border backdrop-blur-sm ${
                message.type === "success"
                  ? "bg-green-500/20 border-green-400/50 text-green-100"
                  : message.type === "error"
                    ? "bg-red-500/20 border-red-400/50 text-red-100"
                    : "bg-blue-500/20 border-blue-400/50 text-blue-100"
              }`}
            >
              <div className="flex items-center space-x-2">
                {message.type === "success" ? (
                  <CheckCircle className="h-4 w-4" />
                ) : (
                  <AlertCircle className="h-4 w-4" />
                )}
                <p className="text-sm">{message.text}</p>
              </div>
            </div>
          )}

          {/* Email Field */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Mail className="h-5 w-5 text-gray-400" />
            </div>
            <Input
              type="email"
              placeholder="EMAIL"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="pl-12 h-12 bg-white/20 backdrop-blur-sm border-white/30 text-white placeholder:text-gray-300 placeholder:uppercase placeholder:tracking-wider placeholder:text-sm focus:bg-white/25 focus:border-white/50"
              required
            />
          </div>

          {/* Password Field (not for forgot password) */}
          {!showForgotPassword && (
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" />
              </div>
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="PASSWORD"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-12 pr-12 h-12 bg-white/20 backdrop-blur-sm border-white/30 text-white placeholder:text-gray-300 placeholder:uppercase placeholder:tracking-wider placeholder:text-sm focus:bg-white/25 focus:border-white/50"
                required
              />
              <button
                type="button"
                className="absolute inset-y-0 right-0 pr-4 flex items-center"
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

          {/* Submit Button */}
          <Button
            type="submit"
            disabled={loading}
            className="w-full h-12 bg-gradient-to-r from-blue-400 to-blue-500 hover:from-blue-500 hover:to-blue-600 text-white font-semibold uppercase tracking-wider shadow-lg transition-all duration-200"
          >
            {loading
              ? "PLEASE WAIT..."
              : showForgotPassword
                ? "SEND RESET EMAIL"
                : "LOGIN"}
          </Button>

          {/* Forgot Password Link */}
          {!showForgotPassword && (
            <div className="text-center">
              <button
                type="button"
                onClick={() => setShowForgotPassword(true)}
                className="text-white/70 hover:text-white italic transition-colors text-sm"
              >
                Forgot your password?
              </button>
            </div>
          )}

          {showForgotPassword && (
            <div className="text-center">
              <button
                type="button"
                onClick={() => setShowForgotPassword(false)}
                className="text-white/70 hover:text-white italic transition-colors text-sm"
              >
                Back to Login
              </button>
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
