import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import AmiAvatar from "@/components/AmiAvatar";
import api from "@/lib/api";

const Register = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      toast({
        title: "Lỗi",
        description: "Mật khẩu xác nhận không khớp",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      await api.post('/register', {
        username: formData.name,
        email: formData.email,
        password: formData.password,
      });

      toast({
        title: "Đăng ký thành công",
        description: "Vui lòng đăng nhập để tiếp tục",
      });
      navigate("/login");
    } catch (error) {
      toast({
        title: "Đăng ký thất bại",
        description: "Vui lòng thử lại sau",
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
            <AmiAvatar size="lg" mood="excited" />
          </div>
          <CardTitle className="text-2xl">Đăng ký ThriveAI</CardTitle>
          <CardDescription>
            Tạo tài khoản để bắt đầu hành trình chăm sóc sức khỏe tinh thần của bạn
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Họ và tên</label>
              <Input
                type="text"
                placeholder="Nguyễn Văn A"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Email</label>
              <Input
                type="email"
                placeholder="email@example.com"
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Mật khẩu</label>
              <Input
                type="password"
                placeholder="••••••••"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
                required
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium">Xác nhận mật khẩu</label>
              <Input
                type="password"
                placeholder="••••••••"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({...formData, confirmPassword: e.target.value})}
                required
              />
            </div>
          </CardContent>
          <CardFooter className="flex flex-col gap-4">
            <Button 
              type="submit" 
              className="w-full" 
              disabled={isLoading}
            >
              {isLoading ? "Đang đăng ký..." : "Đăng ký"}
            </Button>
            <p className="text-sm text-center text-muted-foreground">
              Đã có tài khoản?{" "}
              <Button variant="link" className="p-0" onClick={() => navigate("/login")}>
                Đăng nhập
              </Button>
            </p>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
};

export default Register;