
import { useState, useEffect } from "react";
import { Mic, Video, Headphones, Moon, Sun, Activity } from "lucide-react";
import AmiAvatar from "@/components/AmiAvatar";
import FeatureButton from "@/components/FeatureButton";
import { useIsMobile } from "@/hooks/use-mobile";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";

const Home = () => {
  const isMobile = useIsMobile();
  const [userName, setUserName] = useState("bạn");
  const [showMoodPrompt, setShowMoodPrompt] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  // Check if user has logged mood today
  useEffect(() => {
    // Simulate checking if user has logged mood
    const hasLoggedMood = localStorage.getItem('lastMoodLog');
    const today = new Date().toDateString();
    
    if (!hasLoggedMood || hasLoggedMood !== today) {
      // Show mood prompt after 2 seconds
      const timer = setTimeout(() => {
        setShowMoodPrompt(true);
      }, 2000);
      
      return () => clearTimeout(timer);
    }
  }, []);
  
  // Toggle dark mode
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
    
    // Would also save preference to localStorage in a real app
    toast({
      title: isDarkMode ? "Chế độ sáng đã bật" : "Chế độ tối đã bật",
      description: "Thay đổi giao diện để phù hợp với thời gian trong ngày",
    });
  };
  
  // Handle user mood input
  const handleMoodLog = (mood: string) => {
    localStorage.setItem('lastMoodLog', new Date().toDateString());
    localStorage.setItem('lastMood', mood);
    setShowMoodPrompt(false);
    
    toast({
      title: "Cảm ơn bạn!",
      description: "Ami đã ghi nhận tâm trạng của bạn hôm nay.",
    });
  };

  return (
    <div className="space-y-8 py-4 md:py-8">
      {/* Dark mode toggle */}
      <div className="absolute top-4 right-4 md:top-6 md:right-6">
        <Button 
          variant="outline" 
          size="icon" 
          className="rounded-full"
          onClick={toggleDarkMode}
        >
          {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </Button>
      </div>
      
      {/* Hero section with Ami */}
      <section className="text-center space-y-6">
        <div className="flex flex-col items-center">
          <AmiAvatar 
            size={isMobile ? "lg" : "xl"} 
            mood="happy" 
            withGlow={true} 
          />
          
          <div className="mt-8 space-y-4">
            <h1 className="text-2xl md:text-4xl font-bold">
              Xin chào, {userName}!
            </h1>
            <p className="text-lg text-muted-foreground max-w-lg mx-auto">
              Người bạn đồng hành giúp bạn cảm thấy khỏe mạnh hơn về tinh thần mỗi ngày.
            </p>
            
            <div className="pt-4">
              <Button className="thrive-button text-lg" asChild>
                <a href="/chat">
                  Trò chuyện với Ami
                </a>
              </Button>
            </div>
          </div>
        </div>
      </section>
      
      {/* Mood tracking prompt */}
      {showMoodPrompt && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md p-6">
            <h2 className="text-xl font-bold mb-4">Bạn cảm thấy thế nào hôm nay?</h2>
            <p className="text-muted-foreground mb-6">
              Chia sẻ tâm trạng giúp Ami hiểu bạn tốt hơn và đưa ra những gợi ý phù hợp.
            </p>
            
            <div className="grid grid-cols-5 gap-2 mb-4">
              <Button 
                variant="outline" 
                className="flex flex-col p-3" 
                onClick={() => handleMoodLog("very_happy")}
              >
                <span className="text-2xl">😄</span>
                <span className="text-xs mt-1">Tuyệt vời</span>
              </Button>
              <Button 
                variant="outline" 
                className="flex flex-col p-3" 
                onClick={() => handleMoodLog("happy")}
              >
                <span className="text-2xl">🙂</span>
                <span className="text-xs mt-1">Tốt</span>
              </Button>
              <Button 
                variant="outline" 
                className="flex flex-col p-3" 
                onClick={() => handleMoodLog("neutral")}
              >
                <span className="text-2xl">😐</span>
                <span className="text-xs mt-1">Bình thường</span>
              </Button>
              <Button 
                variant="outline" 
                className="flex flex-col p-3" 
                onClick={() => handleMoodLog("sad")}
              >
                <span className="text-2xl">🙁</span>
                <span className="text-xs mt-1">Buồn</span>
              </Button>
              <Button 
                variant="outline" 
                className="flex flex-col p-3" 
                onClick={() => handleMoodLog("very_sad")}
              >
                <span className="text-2xl">😢</span>
                <span className="text-xs mt-1">Rất tệ</span>
              </Button>
            </div>
            
            <div className="flex justify-between">
              <Button 
                variant="ghost" 
                onClick={() => setShowMoodPrompt(false)}
              >
                Để sau
              </Button>
              <Button onClick={() => handleMoodLog("skip")}>
                Tiếp tục
              </Button>
            </div>
          </Card>
        </div>
      )}
      
      {/* Main features */}
      <section className="pt-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
          <FeatureButton
            to="/chat"
            icon={<Mic />}
            label="Nói chuyện với Ami"
            description="Trò chuyện về cảm xúc và nhận hỗ trợ"
            className="bg-gradient-to-r from-thrive-lavender to-thrive-blue"
          />
          <FeatureButton
            to="/video-call"
            icon={<Video />}
            label="Gọi video"
            description="Kết nối trực tiếp với Ami hoặc chuyên gia"
            className="bg-gradient-to-r from-thrive-blue to-thrive-teal"
          />
          <FeatureButton
            to="/podcast"
            icon={<Headphones />}
            label="Podcast & Video"
            description="Nội dung động lực cho tinh thần"
            className="bg-gradient-to-r from-thrive-teal to-thrive-peach"
          />
        </div>
      </section>
      
      {/* Wellness tip of the day */}
      <section className="pt-6">
        <div className="thrive-card p-6 border border-thrive-lavender">
          <h3 className="font-medium text-lg mb-2">Lời khuyên hôm nay</h3>
          <p className="text-muted-foreground">
            "Hãy dành 5 phút mỗi ngày để nghỉ ngơi và thư giãn. Nhắm mắt, hít thở sâu và cảm nhận sự bình yên trong tâm hồn."
          </p>
        </div>
      </section>
      
      {/* Recent mood tracking */}
      <section className="pt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Tâm trạng gần đây</h2>
          <Button variant="link" className="font-medium" asChild>
            <a href="/mood-tracker">Xem đầy đủ</a>
          </Button>
        </div>
        
        <div className="thrive-card p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Hôm nay</p>
              <div className="flex items-center mt-1">
                <span className="text-2xl mr-2">😊</span>
                <span>Tốt</span>
              </div>
            </div>
            
            <Button variant="outline" size="sm" className="text-primary" asChild>
              <a href="/mood-tracker">
                <Activity className="mr-2 h-4 w-4" /> Cập nhật
              </a>
            </Button>
          </div>
          
          <div className="mt-4 h-24 flex items-end justify-between px-2">
            {['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN'].map((day, i) => {
              // Simulated mood values (would come from API in real app)
              const heights = [40, 60, 50, 30, 45, 70, 80];
              return (
                <div key={day} className="flex flex-col items-center">
                  <div 
                    className="w-6 bg-primary rounded-t-sm" 
                    style={{height: heights[i] + '%'}}
                  ></div>
                  <span className="text-xs mt-1">{day}</span>
                </div>
              );
            })}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
