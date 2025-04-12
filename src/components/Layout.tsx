import { useState, useEffect } from "react";
import { Outlet, NavLink, useLocation } from "react-router-dom";
import { Home, MessageCircle, Video, BarChart3, BookOpen, Menu, X, Settings } from "lucide-react";
import { useIsMobile } from "@/hooks/use-mobile";
import { cn } from "@/lib/utils";
import { Drawer, DrawerContent, DrawerTrigger } from "@/components/ui/drawer";
import ChatButton from "@/components/ChatButton";

const Layout = () => {
  const isMobile = useIsMobile();
  const location = useLocation();
  const [title, setTitle] = useState("ThriveAI");
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  
  useEffect(() => {
    // Set page title based on current route
    switch (location.pathname) {
      case "/":
        setTitle("ThriveAI");
        break;
      case "/chat":
        setTitle("Trò chuyện với Ami");
        break;
      case "/video-call":
        setTitle("Gọi video");
        break;
      case "/journal":
        setTitle("Nhật ký cảm xúc");
        break;
      case "/mood-tracker":
        setTitle("Biểu đồ tâm trạng");
        break;
      case "/podcast":
        setTitle("Podcast & Video");
        break;
      case "/settings":
        setTitle("Cài đặt");
        break;
      default:
        setTitle("ThriveAI");
    }
  }, [location.pathname]);

  // Main 4 mobile nav items
  const mainNavItems = [
    { to: "/", icon: <Home size={24} />, label: "Trang chủ" },
    { to: "/chat", icon: <MessageCircle size={24} />, label: "Trò chuyện" },
    { to: "/video-call", icon: <Video size={24} />, label: "Gọi video" },
    // { to: "/mood-tracker", icon: <BarChart3 size={24} />, label: "Tâm trạng" },
  ];

  // Additional items for the slide-out drawer menu
  const menuItems = [
    // { to: "/journal", icon: <BookOpen size={24} />, label: "Nhật ký" },
    // { to: "/podcast", icon: <BookOpen size={24} />, label: "Thư viện" },
    // { to: "/settings", icon: <Settings size={24} />, label: "Cài đặt" },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header for mobile */}
      {isMobile && (
        <header className="fixed top-0 left-0 right-0 h-16 bg-white shadow-sm z-50 flex items-center justify-between px-4">
          <h1 className="text-xl font-bold">{title}</h1>
          
          <Drawer>
            <DrawerTrigger asChild>
              <button className="p-2 rounded-full hover:bg-muted">
                <Menu size={24} />
              </button>
            </DrawerTrigger>
            <DrawerContent>
              <div className="p-4">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-lg font-bold">Menu</h2>
                </div>
                <nav>
                  <ul className="space-y-4">
                    {menuItems.map((item) => (
                      <li key={item.to}>
                        <NavLink
                          to={item.to}
                          className={({ isActive }) =>
                            cn(
                              "flex items-center gap-3 px-4 py-3 rounded-lg transition-all",
                              isActive
                                ? "bg-accent text-primary font-medium"
                                : "hover:bg-muted"
                            )
                          }
                        >
                          {item.icon}
                          <span>{item.label}</span>
                        </NavLink>
                      </li>
                    ))}
                  </ul>
                </nav>
              </div>
            </DrawerContent>
          </Drawer>
        </header>
      )}

      {/* Sidebar for desktop */}
      {!isMobile && (
        <aside className="fixed left-0 top-0 bottom-0 w-64 bg-white shadow-md z-50 p-6 flex flex-col">
          <div className="flex items-center gap-3 mb-10">
            <div className="w-10 h-10 rounded-full bg-thrive-gradient flex items-center justify-center">
              <span className="text-white font-bold text-lg">T</span>
            </div>
            <h1 className="text-xl font-bold">ThriveAI</h1>
          </div>
          
          <nav className="flex-1">
            <ul className="space-y-2">
              {[...mainNavItems, ...menuItems].map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    className={({ isActive }) =>
                      cn(
                        "flex items-center gap-3 px-4 py-3 rounded-lg transition-all",
                        isActive
                          ? "bg-accent text-primary font-medium"
                          : "hover:bg-muted"
                      )
                    }
                  >
                    {item.icon}
                    <span>{item.label}</span>
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          
          <div className="thrive-card mt-auto">
            <p className="text-sm text-muted-foreground">Bạn cần hỗ trợ?</p>
            <button className="w-full mt-2 text-sm font-medium text-primary">
              Liên hệ chuyên gia
            </button>
          </div>
        </aside>
      )}

      {/* Main content */}
      <main className={cn(
        "flex-1",
        isMobile ? "pt-16 pb-20" : "pl-64"
      )}>
        <div className="container max-w-5xl mx-auto p-4 md:p-6">
          <Outlet />
        </div>
      </main>

      {/* Mobile navigation bar - limit to 4 main items */}
      {isMobile && (
        <nav className="fixed bottom-0 left-0 right-0 h-16 bg-white shadow-lg z-50 flex items-center justify-around">
          {mainNavItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "flex flex-col items-center justify-center w-16 h-16 transition-all",
                  isActive ? "text-primary" : "text-muted-foreground"
                )
              }
            >
              {item.icon}
              <span className="text-xs mt-1">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      )}

      {/* Floating chat button - visible on all pages */}
      <ChatButton />
    </div>
  );
};

export default Layout;
