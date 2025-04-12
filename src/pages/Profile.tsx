import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { toast } from "@/components/ui/use-toast";

interface UserProfile {
  id: string;
  name: string;
  email: string;
  provider?: string;
}

const Profile = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState<UserProfile | null>(null);

  useEffect(() => {
    const userStr = localStorage.getItem("user");
    if (userStr) {
      setUser(JSON.parse(userStr));
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("user");
    toast({
      title: "Đăng xuất thành công",
      description: "Hẹn gặp lại bạn!",
    });
    navigate("/login");
  };

  if (!user) return null;

  return (
    <div className="container mx-auto p-4">
      <Card className="max-w-md mx-auto">
        <CardHeader>
          <CardTitle>Thông tin tài khoản</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center space-x-4">
            <Avatar className="h-20 w-20">
              <AvatarImage src={`https://api.dicebear.com/7.x/initials/svg?seed=${user.name}`} />
              <AvatarFallback>{user.name.substring(0, 2).toUpperCase()}</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="text-xl font-medium">{user.name}</h3>
              <p className="text-sm text-muted-foreground">{user.email}</p>
              {user.provider && (
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                  {user.provider === 'google' ? 'Google Account' : 'Email Account'}
                </span>
              )}
            </div>
          </div>

          <div className="pt-4">
            <Button 
              variant="destructive" 
              className="w-full"
              onClick={handleLogout}
            >
              Đăng xuất
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Profile;