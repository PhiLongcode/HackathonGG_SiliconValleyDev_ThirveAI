
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import Chat from "./pages/Chat";
import VideoCall from "./pages/VideoCall";
import Journal from "./pages/Journal";
import MoodTracker from "./pages/MoodTracker";
import Podcast from "./pages/Podcast";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";
import Login from "./pages/Login";
import Register from "./pages/Register";
import PrivateRoute from "./components/PrivateRoute";
import { GoogleOAuthProvider } from '@react-oauth/google';

// Add imports
import Profile from "./pages/Profile";
import Admin from "./pages/Admin";
import Doctor from "./pages/Doctor";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="chat" element={<Chat />} />
              {/* Update VideoCall routes to support call types */}
              <Route path="video-call" element={<VideoCall />} />
              <Route path="video-call/:callType" element={<VideoCall />} />
              <Route path="journal" element={<Journal />} />
              <Route path="mood-tracker" element={<MoodTracker />} />
              <Route path="podcast" element={<Podcast />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
          <Toaster />
          <Sonner />
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
