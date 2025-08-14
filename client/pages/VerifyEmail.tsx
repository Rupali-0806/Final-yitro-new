import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "@/components/RealAuthProvider";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle, Mail, Loader2 } from "lucide-react";

export default function VerifyEmail() {
  const [searchParams] = useSearchParams();
  const [verificationStatus, setVerificationStatus] = useState<
    "loading" | "success" | "error"
  >("loading");
  const [message, setMessage] = useState("");
  const navigate = useNavigate();
  const { verifyEmail } = useAuth();

  useEffect(() => {
    const token = searchParams.get("token");
    if (token) {
      handleVerification(token);
    } else {
      setVerificationStatus("error");
      setMessage(
        "Invalid verification link. Please check your email for the correct link.",
      );
    }
  }, [searchParams]);

  const handleVerification = async (token: string) => {
    try {
      const result = await verifyEmail(token);
      if (result.success) {
        setVerificationStatus("success");
        setMessage(result.message || "Email verified successfully!");
        // Redirect to dashboard after 3 seconds
        setTimeout(() => {
          navigate("/dashboard");
        }, 3000);
      } else {
        setVerificationStatus("error");
        setMessage(
          result.message || "Failed to verify email. Please try again.",
        );
      }
    } catch (error) {
      setVerificationStatus("error");
      setMessage("An error occurred during verification. Please try again.");
    }
  };

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
        </div>

        {/* Verification Status */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 text-center">
          {verificationStatus === "loading" && (
            <>
              <Loader2 className="mx-auto h-16 w-16 text-blue-500 animate-spin mb-4" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Verifying Your Email
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Please wait while we verify your email address...
              </p>
            </>
          )}

          {verificationStatus === "success" && (
            <>
              <CheckCircle className="mx-auto h-16 w-16 text-green-500 mb-4" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Email Verified! 🎉
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mb-4">{message}</p>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                You will be automatically redirected to your dashboard in a few
                seconds...
              </p>
              <Button
                onClick={() => navigate("/dashboard")}
                className="bg-green-600 hover:bg-green-700 text-white"
              >
                Go to Dashboard Now
              </Button>
            </>
          )}

          {verificationStatus === "error" && (
            <>
              <XCircle className="mx-auto h-16 w-16 text-red-500 mb-4" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                Verification Failed
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mb-6">{message}</p>
              <div className="space-y-3">
                <Button
                  onClick={() => navigate("/login")}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                >
                  Back to Login
                </Button>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Need help? Contact our support team.
                </p>
              </div>
            </>
          )}
        </div>

        {/* Help Section */}
        {verificationStatus !== "loading" && (
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center">
            <Mail className="mx-auto h-8 w-8 text-white/70 mb-2" />
            <p className="text-white/90 text-sm">
              Didn't receive the verification email? Check your spam folder or
              contact support.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
