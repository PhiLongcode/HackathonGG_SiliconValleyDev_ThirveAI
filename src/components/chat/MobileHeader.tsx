
import AmiAvatar from "@/components/AmiAvatar";

interface MobileHeaderProps {
  amiMood: "happy" | "neutral" | "thinking" | "sad" | "excited";
  isSpeaking: boolean;
}

const MobileHeader = ({ amiMood, isSpeaking }: MobileHeaderProps) => {
  return (
    <div className="flex items-center p-4 border-b">
      <AmiAvatar size="sm" mood={amiMood} className="mr-4" />
      <div>
        <h2 className="font-medium">Ami</h2>
        <p className="text-xs text-muted-foreground">
          {amiMood === "thinking" ? "Đang suy nghĩ..." : 
           isSpeaking ? "Đang nói..." : "Đang hoạt động"}
        </p>
      </div>
    </div>
  );
};

export default MobileHeader;
