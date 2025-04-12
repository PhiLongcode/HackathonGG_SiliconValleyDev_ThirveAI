
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import AmiAvatar from "@/components/AmiAvatar";
import { GoogleLogin } from '@react-oauth/google';
import api from "@/lib/api";

const Login = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await api.post('/token', new URLSearchParams({
        username: email,
        password: password
      }), {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });

      const { access_token } = response.data;
      localStorage.setItem("token", access_token);

      // Determine user role based on email (in a real app, this would come from the backend)
      let userRole = "user";
      if (email.includes("admin")) {
        userRole = "admin";
      } else if (email.includes("doctor")) {
        userRole = "doctor";
      }

      // Store user info with role
      localStorage.setItem("user", JSON.stringify({
        id: access_token,
        name: email.split('@')[0],
        email: email,
        isLoggedIn: true,
        role: userRole
      }));

      toast({
        title: "Đăng nhập thành công",
        description: "Chào mừng bạn đến với ThriveAI!",
      });

      // Redirect based on role
      if (userRole === "admin") {
        navigate("/admin");
      } else if (userRole === "doctor") {
        navigate("/doctor");
      } else {
        navigate("/");
      }
    } catch (error) {
      toast({
        title: "Đăng nhập thất bại",
        description: "Email hoặc mật khẩu không đúng",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse: any) => {
    try {
      // Simulate API call to verify Google token
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Store user info with default role
      localStorage.setItem("user", JSON.stringify({
        id: credentialResponse.clientId,
        name: "Google User",
        email: "user@gmail.com",
        isLoggedIn: true,
        provider: "google",
        role: "user"
      }));

      toast({
        title: "Đăng nhập thành công",
        description: "Chào mừng bạn đến với ThriveAI!",
      });

      navigate("/");
    } catch (error) {
      toast({
        title: "Đăng nhập thất bại",
        description: "Không thể đăng nhập với Google",
        variant: "destructive",
      });
    }
  };

  // Add sample account login functions for different roles
  const handleUserLogin = () => {
    setEmail("user@thriveai.com");
    setPassword("user123");
    loginWithRole("user");
  };

  const handleDoctorLogin = () => {
    setEmail("doctor@thriveai.com");
    setPassword("doctor123");
    loginWithRole("doctor");
  };

  const handleAdminLogin = () => {
    setEmail("admin@thriveai.com");
    setPassword("admin123");
    loginWithRole("admin");
  };

  const loginWithRole = async (role: string) => {
    setIsLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      localStorage.setItem("user", JSON.stringify({
        id: "demo-" + role,
        name: role === "user" ? "Người Dùng Demo" : role === "doctor" ? "Bác Sĩ Demo" : "Admin Demo",
        email: `${role}@thriveai.com`,
        isLoggedIn: true,
        role: role
      }));

      toast({
        title: "Đăng nhập thành công",
        description: "Chào mừng bạn đến với ThriveAI!",
      });

      // Redirect based on role
      if (role === "admin") {
        navigate("/admin");
      } else if (role === "doctor") {
        navigate("/doctor");
      } else {
        navigate("/");
      }
    } catch (error) {
      toast({
        title: "Đăng nhập thất bại",
        description: "Có lỗi xảy ra",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-thrive-lavender/20 to-thrive-blue/20 p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <AmiAvatar size="lg" mood="happy" />
          </div>
          <CardTitle className="text-2xl">Đăng nhập ThriveAI</CardTitle>
          <CardDescription>
            Chào mừng trở lại! Hãy đăng nhập để tiếp tục trò chuyện với Ami
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleLogin}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Email</label>
              <Input
                type="email"
                placeholder="email@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Mật khẩu</label>
              <Input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">
                  Hoặc đăng nhập với
                </span>
              </div>
            </div>

            <div className="flex justify-center">
              <GoogleLogin
                onSuccess={handleGoogleSuccess}
                onError={() => {
                  toast({
                    title: "Đăng nhập thất bại",
                    description: "Không thể đăng nhập với Google",
                    variant: "destructive",
                  });
                }}
              />
            </div>
            <div className="flex flex-wrap justify-center gap-2 mt-4">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={handleUserLogin}
              >
                Tài khoản người dùng
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={handleDoctorLogin}
              >
                Tài khoản bác sĩ
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="text-xs"
                onClick={handleAdminLogin}
              >
                Tài khoản admin
              </Button>
            </div>
          </CardContent>
          <CardFooter className="flex flex-col gap-4">
            <Button 
              type="submit" 
              className="w-full" 
              disabled={isLoading}
            >
              {isLoading ? "Đang đăng nhập..." : "Đăng nhập"}
            </Button>
            <p className="text-sm text-center text-muted-foreground">
              Chưa có tài khoản?{" "}
              <Button variant="link" className="p-0" onClick={() => navigate("/register")}>
                Đăng ký ngay
              </Button>
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
};

export default Login;
