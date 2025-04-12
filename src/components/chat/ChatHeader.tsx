
import { cn } from "@/lib/utils";
import AmiAvatar from "@/components/AmiAvatar";
import { Button } from "@/components/ui/button";
import { ArrowUp } from "lucide-react";

interface ChatHeaderProps {
  amiMood: "happy" | "neutral" | "thinking" | "sad" | "excited";
  isSpeaking: boolean;
  onClose: () => void;
}

const ChatHeader = ({ amiMood, isSpeaking, onClose }: ChatHeaderProps) => {
  return (
    <div className="flex items-center justify-between p-4 border-b">
      <div className="flex items-center">
        <AmiAvatar size="sm" mood={amiMood} className="mr-4" />
        <div>
          <h2 className="font-medium">Ami</h2>
          <p className="text-xs text-muted-foreground">
            {amiMood === "thinking" ? "Đang suy nghĩ..." : 
            isSpeaking ? "Đang nói..." : "Đang hoạt động"}
          </p>
        </div>
      </div>
      <Button variant="ghost" size="icon" onClick={onClose}>
        <ArrowUp className="h-4 w-4" />
      </Button>
    </div>
  );
};

export default ChatHeader;
