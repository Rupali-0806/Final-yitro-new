import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

interface User {
  id: string;
  email: string;
  displayName?: string;
  role: "admin" | "user";
  primaryEmail: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<boolean>;
  signUp: (
    email: string,
    password: string,
    displayName: string,
  ) => Promise<boolean>;
  signOut: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing session
    const checkSession = () => {
      const storedUser = localStorage.getItem("crm_user");
      if (storedUser) {
        try {
          const userData = JSON.parse(storedUser);
          // Clear old yutro domain sessions
          if (userData.email && userData.email.includes("yutro.com")) {
            localStorage.removeItem("crm_user");
            setLoading(false);
            return;
          }
          setUser(userData);
        } catch (error) {
          console.error("Error parsing stored user:", error);
          localStorage.removeItem("crm_user");
        }
      }
      setLoading(false);
    };

    checkSession();
  }, []);

  const determineUserRole = (email: string): "admin" | "user" => {
    // Admin users: emails containing "admin" or specific admin emails
    return email.includes("admin") || email === "admin@yitro.com"
      ? "admin"
      : "user";
  };

  const signIn = async (email: string, password: string): Promise<boolean> => {
    try {
      // For demo purposes, accept any email/password combination
      // In production, this would validate against Neon Auth
      const role = determineUserRole(email);

      const userData: User = {
        id: `user_${Date.now()}`,
        email,
        primaryEmail: email,
        displayName: email.split("@")[0],
        role,
      };

      setUser(userData);
      localStorage.setItem("crm_user", JSON.stringify(userData));
      return true;
    } catch (error) {
      console.error("Sign in error:", error);
      return false;
    }
  };

  const signUp = async (
    email: string,
    password: string,
    displayName: string,
  ): Promise<boolean> => {
    try {
      // For demo purposes, create user immediately
      // In production, this would create user via Neon Auth
      const role = determineUserRole(email);

      const userData: User = {
        id: `user_${Date.now()}`,
        email,
        primaryEmail: email,
        displayName: displayName || email.split("@")[0],
        role,
      };

      setUser(userData);
      localStorage.setItem("crm_user", JSON.stringify(userData));
      return true;
    } catch (error) {
      console.error("Sign up error:", error);
      return false;
    }
  };

  const signOut = () => {
    setUser(null);
    localStorage.removeItem("crm_user");
  };

  const value: AuthContextType = {
    user,
    loading,
    signIn,
    signUp,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Compatibility hook for Stack Auth pattern
export function useUser() {
  const { user } = useAuth();
  return user;
}
