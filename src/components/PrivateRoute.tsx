
import { Navigate, useLocation } from "react-router-dom";

type Role = "user" | "doctor" | "admin";

interface PrivateRouteProps {
  children: React.ReactNode;
  requiredRole?: Role;
}

const PrivateRoute = ({ children, requiredRole }: PrivateRouteProps) => {
  const location = useLocation();
  const user = JSON.parse(localStorage.getItem("user") || "null");

  if (!user?.isLoggedIn) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // If a specific role is required, check if the user has it
  if (requiredRole && user.role !== requiredRole) {
    // Redirect users to their appropriate dashboard based on their role
    if (user.role === "admin") {
      return <Navigate to="/admin" replace />;
    } else if (user.role === "doctor") {
      return <Navigate to="/doctor" replace />;
    } else {
      return <Navigate to="/" replace />;
    }
  }

  return children;
};

export default PrivateRoute;
