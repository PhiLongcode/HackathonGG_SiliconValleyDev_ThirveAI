
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, Moon, Volume2, Lock, Globe, UserCircle, LogOut } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "@/components/ui/use-toast";
import AmiAvatar from "@/components/AmiAvatar";

const Settings = () => {
  const [volume, setVolume] = useState(75);
  const [language, setLanguage] = useState("vi");
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("user");
    toast({
      title: "Đăng xuất thành công",
      description: "Hẹn gặp lại bạn!",
    });
    navigate("/login");
  };

  // Get user data from localStorage
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const userInitials = user.name ? user.name.split(" ").map((n: string) => n[0]).join("") : "U";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Cài đặt</h1>
        <p className="text-muted-foreground">
          Điều chỉnh các tùy chọn của ứng dụng
        </p>
      </div>
      
      <div className="space-y-6">
        {/* Profile section with dynamic data */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <UserCircle className="h-5 w-5" />
              Tài khoản
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col sm:flex-row items-center gap-6">
            <div className="w-24 h-24 rounded-full bg-primary flex items-center justify-center text-3xl text-white font-medium">
              {userInitials}
            </div>
            <div>
              <h3 className="font-medium text-lg">{user.name || "Người dùng"}</h3>
              <p className="text-muted-foreground">{user.email || "email@example.com"}</p>
              <Button variant="outline" className="mt-3">Chỉnh sửa hồ sơ</Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Ami settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Cài đặt Ami
            </CardTitle>
            <CardDescription>
              Tùy chỉnh người bạn AI của bạn
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex flex-col sm:flex-row items-center gap-6">
              <AmiAvatar size="md" />
              <div className="flex-1">
                <h3 className="font-medium mb-2">Giọng nói Ami</h3>
                <div className="flex items-center gap-4">
                  <Volume2 className="h-5 w-5 text-muted-foreground" />
                  <Slider
                    value={[volume]}
                    max={100}
                    step={1}
                    className="flex-1"
                    onValueChange={(value) => setVolume(value[0])}
                  />
                  <span className="w-10 text-right">{volume}%</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Trả lời tự động</Label>
                  <p className="text-sm text-muted-foreground">
                    Ami sẽ gửi lời chào khi bạn mở ứng dụng
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Gợi ý hàng ngày</Label>
                  <p className="text-sm text-muted-foreground">
                    Nhận gợi ý để cải thiện sức khỏe tinh thần
                  </p>
                </div>
                <Switch defaultChecked />
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* App settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Cài đặt ứng dụng
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5" />
                <div>
                  <Label>Thông báo</Label>
                  <p className="text-sm text-muted-foreground">
                    Nhận thông báo từ ứng dụng
                  </p>
                </div>
              </div>
              <Switch defaultChecked />
            </div>
            
            <Separator />
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Moon className="h-5 w-5" />
                <div>
                  <Label>Chế độ tối</Label>
                  <p className="text-sm text-muted-foreground">
                    Thay đổi giao diện sang tối
                  </p>
                </div>
              </div>
              <Switch />
            </div>
            
            <Separator />
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Globe className="h-5 w-5" />
                <div>
                  <Label>Ngôn ngữ</Label>
                  <p className="text-sm text-muted-foreground">
                    Thay đổi ngôn ngữ hiển thị
                  </p>
                </div>
              </div>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Chọn ngôn ngữ" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="vi">Tiếng Việt</SelectItem>
                  <SelectItem value="en">English</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <Separator />
            
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Lock className="h-5 w-5" />
                <div>
                  <Label>Quyền riêng tư</Label>
                  <p className="text-sm text-muted-foreground">
                    Quản lý dữ liệu và quyền riêng tư
                  </p>
                </div>
              </div>
              <Button variant="outline" size="sm">
                Quản lý
              </Button>
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              variant="outline" 
              className="w-full flex items-center gap-2 text-red-500 hover:text-red-600"
              onClick={handleLogout}
            >
              <LogOut className="h-4 w-4" /> Đăng xuất
            </Button>
          </CardFooter>
        </Card>
        
        {/* About section */}
        <Card>
          <CardHeader>
            <CardTitle>Thông tin ứng dụng</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Phiên bản</span>
              <span>1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Cập nhật gần nhất</span>
              <span>06/04/2025</span>
            </div>
          </CardContent>
          <CardFooter className="flex flex-col gap-2">
            <Button variant="link" className="w-full justify-start p-0">
              Chính sách bảo mật
            </Button>
            <Button variant="link" className="w-full justify-start p-0">
              Điều khoản sử dụng
            </Button>
          </CardFooter>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="text-red-600 flex items-center gap-2">
            <LogOut className="h-5 w-5" />
            Đăng xuất
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button 
            variant="destructive" 
            className="w-full"
            onClick={handleLogout}
          >
            Đăng xuất khỏi ThriveAI
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export default Settings;
