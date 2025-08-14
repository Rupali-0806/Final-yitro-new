import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/components/AuthProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { User, Lock, Mail, UserPlus } from "lucide-react";

export default function Login() {
  const [showCustomLogin, setShowCustomLogin] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [rememberMe, setRememberMe] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const { user, signIn, signUp } = useAuth();

  // Redirect to dashboard if user is already authenticated
  useEffect(() => {
    if (user) {
      navigate("/dashboard");
    }
  }, [user, navigate]);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError("Please fill in all fields");
      return;
    }

    setLoading(true);
    setError("");

    try {
      let success;
      if (isSignUp) {
        success = await signUp(email, password, displayName);
      } else {
        success = await signIn(email, password);
      }

      if (success) {
        navigate("/dashboard");
      } else {
        setError("Authentication failed. Please try again.");
      }
    } catch (error) {
      setError("An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleCustomLogin = (e: React.FormEvent) => {
    e.preventDefault();
    // For demo purposes, navigate directly (in production, validate with Stack Auth)
    navigate("/dashboard");
  };

  const handleQuickDemo = (userType: "admin" | "user") => {
    const demoEmail =
      userType === "admin" ? "admin@yitro.com" : "user@yitro.com";
    signIn(demoEmail, "demo123").then(() => {
      navigate("/dashboard");
    });
  };

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

        {/* Quick Demo Buttons */}
        <div className="space-y-3">
          <Button
            onClick={() => handleQuickDemo("admin")}
            className="w-full h-12 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-semibold uppercase tracking-wider shadow-lg transition-all duration-200"
          >
            🔑 Admin
          </Button>

          <Button
            onClick={() => handleQuickDemo("user")}
            className="w-full h-12 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-semibold uppercase tracking-wider shadow-lg transition-all duration-200"
          >
            👤 User
          </Button>
        </div>

        {/* Divider */}
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-white/30"></div>
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-gradient-to-r from-transparent via-blue-500/20 to-transparent px-4 text-white/70">
              Or create account
            </span>
          </div>
        </div>

        {/* Auth Toggle */}
        <Button
          type="button"
          variant="outline"
          onClick={() => setShowCustomLogin(!showCustomLogin)}
          className="w-full h-10 bg-white/10 backdrop-blur-sm border-white/30 text-white hover:bg-white/20"
        >
          {showCustomLogin ? "Hide Auth Form" : "Create Account / Sign In"}
        </Button>

        {/* Auth Form */}
        {showCustomLogin && (
          <form onSubmit={handleAuth} className="space-y-6 mt-6">
            {error && (
              <div className="p-3 text-sm text-red-200 bg-red-500/20 border border-red-500/30 rounded-lg">
                {error}
              </div>
            )}

            {/* Sign Up Toggle */}
            <div className="flex items-center justify-center space-x-4">
              <button
                type="button"
                onClick={() => setIsSignUp(false)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                  !isSignUp
                    ? "bg-white/20 text-white border border-white/30"
                    : "text-white/70 hover:text-white"
                }`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => setIsSignUp(true)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                  isSignUp
                    ? "bg-white/20 text-white border border-white/30"
                    : "text-white/70 hover:text-white"
                }`}
              >
                Sign Up
              </button>
            </div>

            {/* Display Name Field (Sign Up Only) */}
            {isSignUp && (
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <UserPlus className="h-5 w-5 text-gray-400" />
                </div>
                <Input
                  type="text"
                  placeholder="DISPLAY NAME"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="pl-12 h-12 bg-white/20 backdrop-blur-sm border-white/30 text-white placeholder:text-gray-300 placeholder:uppercase placeholder:tracking-wider placeholder:text-sm focus:bg-white/25 focus:border-white/50"
                />
              </div>
            )}

            {/* Email Field */}
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Mail className="h-5 w-5 text-gray-400" />
              </div>
              <Input
                type="email"
                placeholder="EMAIL ADDRESS"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="pl-12 h-12 bg-white/20 backdrop-blur-sm border-white/30 text-white placeholder:text-gray-300 placeholder:uppercase placeholder:tracking-wider placeholder:text-sm focus:bg-white/25 focus:border-white/50"
                required
              />
            </div>

            {/* Password Field */}
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-gray-400" />
              </div>
              <Input
                type="password"
                placeholder="PASSWORD"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="pl-12 h-12 bg-white/20 backdrop-blur-sm border-white/30 text-white placeholder:text-gray-300 placeholder:uppercase placeholder:tracking-wider placeholder:text-sm focus:bg-white/25 focus:border-white/50"
                required
              />
            </div>

            {/* Auth Button */}
            <Button
              type="submit"
              disabled={loading}
              className="w-full h-12 bg-gradient-to-r from-blue-400 to-blue-500 hover:from-blue-500 hover:to-blue-600 text-white font-semibold uppercase tracking-wider shadow-lg transition-all duration-200 disabled:opacity-50"
            >
              {loading
                ? "PROCESSING..."
                : isSignUp
                  ? "CREATE ACCOUNT"
                  : "SIGN IN"}
            </Button>

            {/* Remember Me & Forgot Password */}
            {!isSignUp && (
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="remember"
                    checked={rememberMe}
                    onCheckedChange={(checked) =>
                      setRememberMe(checked as boolean)
                    }
                    className="border-white/30 data-[state=checked]:bg-white/20 data-[state=checked]:border-white/50"
                  />
                  <label
                    htmlFor="remember"
                    className="text-white/90 cursor-pointer"
                  >
                    Remember me
                  </label>
                </div>
                <button
                  type="button"
                  className="text-white/70 hover:text-white italic transition-colors"
                >
                  Forgot your password?
                </button>
              </div>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
