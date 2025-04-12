import { Button } from "@/components/ui/button";
import { Mic, MicOff, Camera, CameraOff, PhoneOff, Settings } from "lucide-react";
import { AmiVoiceSettings } from "./AmiVoiceSettings";
import { cn } from "@/lib/utils"; // Added missing import

type CallControlsProps = {
  isMicOn: boolean;
  isCameraOn: boolean;
  isListening: boolean;
  callWith: "ami" | "expert" | null;
  onMicToggle: () => void;
  onCameraToggle: () => void;
  onEndCall: () => void;
  onVoiceToggle: () => void;
  amivoice: { gender: string; region: string };
  onVoiceChange: (voice: { gender: string; region: string }) => void;
};

export const CallControls = ({
  isMicOn,
  isCameraOn,
  isListening,
  callWith,
  onMicToggle,
  onCameraToggle,
  onEndCall,
  onVoiceToggle,
  amivoice,
  onVoiceChange
}: CallControlsProps) => {
  return (
    <div className="h-20 bg-background/95 backdrop-blur flex items-center justify-center gap-6">
      {/* Mic Button */}
      <Button
        variant="outline"
        size="icon"
        className={cn(
          "rounded-full w-14 h-14 border-2 hover:bg-primary/10",
          !isMicOn && "bg-destructive/20 text-destructive border-destructive/50"
        )}
        onClick={onMicToggle}
      >
        {isMicOn ? 
          <Mic className="h-6 w-6" /> : 
          <MicOff className="h-6 w-6" />
        }
      </Button>

      {/* Camera Button */}
      <Button
        variant="outline"
        size="icon"
        className={cn(
          "rounded-full w-14 h-14 border-2 hover:bg-primary/10",
          !isCameraOn && "bg-destructive/20 text-destructive border-destructive/50"
        )}
        onClick={onCameraToggle}
      >
        {isCameraOn ? 
          <Camera className="h-6 w-6" /> : 
          <CameraOff className="h-6 w-6" />
        }
      </Button>

      {/* End Call Button */}
      <Button
        variant="destructive"
        size="icon"
        className="rounded-full w-16 h-16 border-2 hover:bg-destructive/90 shadow-lg"
        onClick={onEndCall}
      >
        <PhoneOff className="h-8 w-8" />
      </Button>

      {/* AMI Controls */}
      {callWith === "ami" && (
        <>
          <Button
            variant="outline"
            size="icon"
            className={cn(
              "rounded-full w-14 h-14 border-2 hover:bg-primary/10",
              isListening && "bg-green-500/20 text-green-600 border-green-500/50"
            )}
            onClick={onVoiceToggle}
          >
            <Mic className="h-6 w-6" />
          </Button>
          
          <AmiVoiceSettings
            currentVoice={amivoice}
            onVoiceChange={onVoiceChange}
          />
        </>
      )}
    </div>
  );
};