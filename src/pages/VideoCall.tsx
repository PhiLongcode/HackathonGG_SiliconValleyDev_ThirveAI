import { useState, useRef, useEffect } from "react";
import { Video, Users, User, Phone, PhoneOff, Mic, MicOff, Camera, CameraOff, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import AmiAvatar from "@/components/AmiAvatar";
import { useIsMobile } from "@/hooks/use-mobile";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Drawer, DrawerContent, DrawerTrigger } from "@/components/ui/drawer";
import { Input } from "@/components/ui/input";
import { toast } from "@/components/ui/use-toast";
import { cn } from "@/lib/utils";
import { useNavigate, useParams } from "react-router-dom";
import { Settings } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import axios from "axios";

type VoiceGender = 'male' | 'female';
type VoiceRegion = 'north' | 'central' | 'south';

const VideoCall = () => {
    const isMobile = useIsMobile();
    const navigate = useNavigate();
    const { callType } = useParams<{ callType: string }>();

    const [isInCall, setIsInCall] = useState(false);
    const [callWith, setCallWith] = useState<"ami" | "expert" | null>(null);
    const [isMicOn, setIsMicOn] = useState(true);
    const [isCameraOn, setIsCameraOn] = useState(true);
    const [isConnecting, setIsConnecting] = useState(false);
    const [expertInfo, setExpertInfo] = useState<{ name: string; specialty: string } | null>(null);
    const [isLowConnection, setIsLowConnection] = useState(false);
    const [chatMessage, setChatMessage] = useState("");
    const [chatMessages, setChatMessages] = useState<{ text: string, sender: "user" | "other", timestamp: Date }[]>([]);
    const chatRef = useRef<HTMLDivElement>(null);

    const [listening, setListening] = useState(false);
    const recognitionRef = useRef<any>(null);
    const localVideoRef = useRef<HTMLVideoElement>(null);
    const remoteVideoRef = useRef<HTMLVideoElement>(null);
    const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
    const localStreamRef = useRef<MediaStream | null>(null);

    // Check if we have a call type from URL params
    useEffect(() => {
        if (callType === "ami" || callType === "expert") {
            handleStartCall(callType);
        }
    }, [callType]);

    // Initialize WebRTC when starting a call
    useEffect(() => {
        if (isInCall && !isConnecting) {
            initializeWebRTC();
            initializeSpeechRecognition();
        }

        return () => {
            if (localStreamRef.current) {
                localStreamRef.current.getTracks().forEach(track => track.stop());
            }
            if (peerConnectionRef.current) {
                peerConnectionRef.current.close();
            }
        };
    }, [isInCall, isConnecting]);

    // Initialize speech recognition
    const initializeSpeechRecognition = () => {
        if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            recognition.lang = "vi-VN";
            recognition.interimResults = true;
            recognition.continuous = false;

            recognition.onstart = () => {
                setListening(true);
                toast({
                    title: "Đang lắng nghe...",
                    description: "Hãy nói điều bạn muốn chia sẻ với Ami",
                });
            };

            recognition.onend = () => {
                setListening(false);
            };

            recognition.onresult = async (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');

                if (event.results[0].isFinal) {
                    addChatMessage(transcript, "user");
                    await sendToAI(transcript);
                }
            };

            recognition.onerror = (event) => {
                console.error("Speech recognition error:", event.error);
                setListening(false);
                let errorMessage = "Đã xảy ra lỗi khi nhận dạng giọng nói";
                if (event.error === 'no-speech') {
                    errorMessage = "Không nghe thấy giọng nói, vui lòng thử lại";
                } else if (event.error === 'aborted') {
                    errorMessage = "Nhận dạng giọng nói đã bị hủy";
                } else if (event.error === 'network') {
                    errorMessage = "Lỗi kết nối mạng, vui lòng kiểm tra kết nối internet";
                } else if (event.error === 'not-allowed') {
                    errorMessage = "Không có quyền truy cập microphone, vui lòng kiểm tra quyền truy cập";
                }
                toast({
                    title: "Lỗi nhận dạng giọng nói",
                    description: errorMessage,
                    variant: "destructive",
                });
            };

            recognitionRef.current = recognition;
        } else {
            toast({
                title: "Không hỗ trợ",
                description: "Trình duyệt của bạn không hỗ trợ nhận dạng giọng nói.",
                variant: "destructive",
            });
        }
    };

    const toggleListening = () => {
        if (!recognitionRef.current) {
            initializeSpeechRecognition();
            return;
        }

        if (!listening) {
            try {
                recognitionRef.current.start();
                setTimeout(() => {
                    if (listening) {
                        recognitionRef.current.stop();
                    }
                }, 10000);
            } catch (error) {
                console.error("Error starting speech recognition:", error);
                initializeSpeechRecognition();
                setTimeout(() => {
                    try {
                        recognitionRef.current?.start();
                    } catch (e) {
                        console.error("Failed to restart speech recognition:", e);
                    }
                }, 100);
            }
        } else {
            try {
                recognitionRef.current.stop();
            } catch (error) {
                console.error("Error stopping speech recognition:", error);
            }
        }
    };

    const speak = (text: string) => {
        const synth = window.speechSynthesis;
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "vi-VN";
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        synth.cancel();
        synth.speak(utterance);
    };

    const sendToAI = async (message: string) => {
        try {
            toast({
                title: "Đang xử lý...",
                description: "Ami đang suy nghĩ về câu trả lời",
            });
            const res = await axios.post("http://localhost:8000/process_text", { message }); //config link backend here
            if (res.data && res.data.reply) {
                const aiReply = res.data.reply;
                addChatMessage(aiReply, "other");
                speak(aiReply);
            } else {
                throw new Error("Invalid response format");
            }
        } catch (err) {
            console.error("AI request error:", err);
            if (axios.isAxiosError(err) && err.code === "ECONNREFUSED") {
                toast({
                    title: "Lỗi kết nối",
                    description: "Không thể kết nối đến máy chủ AI. Vui lòng kiểm tra máy chủ đã chạy chưa.",
                    variant: "destructive",
                });
            } else {
                toast({
                    title: "Lỗi xử lý",
                    description: "Không thể xử lý yêu cầu của bạn. Vui lòng thử lại sau.",
                    variant: "destructive",
                });
            }
        }
    };

    const addChatMessage = (text, sender: "user" | "other") => {
        setChatMessages(prev => [
            ...prev,
            { text, sender, timestamp: new Date() }
        ]);
        setTimeout(() => {
            if (chatRef.current) {
                chatRef.current.scrollTop = chatRef.current.scrollHeight;
            }
        }, 100);
    };

    useEffect(() => {
        if (localStreamRef.current) {
            localStreamRef.current.getAudioTracks().forEach(track => {
                track.enabled = isMicOn;
            });
            if (isCameraOn) {
                const videoTracks = localStreamRef.current.getVideoTracks();
                if (videoTracks.length === 0 || videoTracks[0]?.readyState === "ended" || !videoTracks[0]?.enabled) {
                    videoTracks.forEach(track => track.stop());
                    navigator.mediaDevices.getUserMedia({
                        video: {
                            width: { ideal: 1280 },
                            height: { ideal: 720 },
                            facingMode: "user"
                        }
                    })
                        .then(newStream => {
                            const newVideoTrack = newStream.getVideoTracks()[0];
                            if (newVideoTrack) {
                                if (peerConnectionRef.current) {
                                    const senders = peerConnectionRef.current.getSenders();
                                    const videoSender = senders.find(sender =>
                                        sender.track && sender.track.kind === 'video'
                                    );
                                    if (videoSender) {
                                        videoSender.replaceTrack(newVideoTrack);
                                    }
                                }
                                if (localStreamRef.current) {
                                    const oldTracks = localStreamRef.current.getVideoTracks();
                                    oldTracks.forEach(track => {
                                        localStreamRef.current?.removeTrack(track);
                                        track.stop();
                                    });
                                    localStreamRef.current.addTrack(newVideoTrack);
                                    if (localVideoRef.current) {
                                        localVideoRef.current.srcObject = null;
                                        localVideoRef.current.srcObject = localStreamRef.current;
                                    }
                                }
                            }
                        })
                        .catch(error => {
                            console.error("Error restarting camera:", error);
                            toast({
                                title: "Lỗi camera",
                                description: "Không thể kết nối lại với camera",
                                variant: "destructive",
                            });
                        });
                } else {
                    videoTracks.forEach(track => {
                        track.enabled = true;
                    });
                }
            } else {
                localStreamRef.current.getVideoTracks().forEach(track => {
                    track.enabled = false;
                });
            }
        }
    }, [isMicOn, isCameraOn]);

    const initializeWebRTC = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            }).catch(() => {
                return navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    },
                    audio: true
                });
            });

            localStreamRef.current = stream;
            if (localVideoRef.current) {
                localVideoRef.current.srcObject = stream;
            }

            setTimeout(() => {
                setIsConnecting(false);
            }, 1000);

        } catch (error) {
            console.error("Error accessing media devices:", error);
            toast({
                title: "Lỗi thiết bị",
                description: "Không thể truy cập camera hoặc microphone. Vui lòng kiểm tra quyền truy cập thiết bị.",
                variant: "destructive",
            });
            handleEndCall();
        }
    };

    const handleStartCall = (type: "ami" | "expert") => {
        setCallWith(type);
        setIsInCall(true);
        setIsConnecting(true);

        if (type === "expert") {
            setExpertInfo({
                name: "Dr. Nguyễn Văn A",
                specialty: "Tâm lý học lâm sàng"
            });
        }

        setChatMessages([]);
    };

    const handleEndCall = () => {
        if (localStreamRef.current) {
            localStreamRef.current.getTracks().forEach(track => track.stop());
        }
        if (peerConnectionRef.current) {
            peerConnectionRef.current.close();
            peerConnectionRef.current = null;
        }
        setIsInCall(false);
        setCallWith(null);
        setExpertInfo(null);
        navigate("/video-call");
    };

    const handleSendChatMessage = () => {
        if (!chatMessage.trim()) return;
        addChatMessage(chatMessage, "user");
        if (callWith === "ami") {
            sendToAI(chatMessage);
        }
        setChatMessage("");
    };

    const handleToggleMic = () => {
        setIsMicOn(!isMicOn);
    };

    const handleToggleCamera = () => {
        setIsCameraOn(!isCameraOn);
    };

    // If not in a call, show call selection
    if (!isInCall) {
        return (
            <div className="container max-w-5xl py-12 space-y-12">
                <h1 className="text-4xl font-bold text-center text-foreground">Cuộc gọi video</h1>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">
                    <Card className="hover:shadow-xl transition-shadow duration-300 bg-card border border-border">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-3 text-2xl">
                                <AmiAvatar size="sm" mood="happy" />
                                Gọi video với Ami
                            </CardTitle>
                            <CardDescription className="text-muted-foreground">
                                Trò chuyện trực tiếp với trợ lý AI thân thiện của bạn
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="aspect-video bg-muted rounded-xl flex items-center justify-center overflow-hidden">
                                <AmiAvatar size="lg" mood="happy" className="scale-110 transition-shadow duration-300" />
                            </div>
                        </CardContent>
                        <CardFooter>
                            <Button
                                className="w-full bg-primary hover:bg-primary/90 transition-colors duration-200"
                                onClick={() => handleStartCall("ami")}
                            >
                                <Phone className="mr-2 h-5 w-5" /> Bắt đầu cuộc gọi
                            </Button>
                        </CardFooter>
                    </Card>

                    <Card className="hover:shadow-xl transition-shadow duration-300 bg-card border border-border">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-3 text-2xl">
                                <User className="h-6 w-6" />
                                Gọi video với chuyên gia
                            </CardTitle>
                            <CardDescription className="text-muted-foreground">
                                Kết nối trực tiếp với chuyên gia tâm lý của chúng tôi
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="aspect-video bg-muted rounded-xl flex items-center justify-center overflow-hidden">
                                <Users className="h-20 w-20 text-muted-foreground transition-transform duration-300 hover:scale-110" />
                            </div>
                        </CardContent>
                        <CardFooter>
                            <Button
                                className="w-full bg-primary hover:bg-primary/90 transition-colors duration-200"
                                onClick={() => handleStartCall("expert")}
                            >
                                <Phone className="mr-2 h-5 w-5" /> Bắt đầu cuộc gọi
                            </Button>
                        </CardFooter>
                    </Card>
                </div>
            </div>
        );
    }

    // In call UI
    return (
        <div className="h-[calc(100vh-4rem)] flex flex-col bg-background">
            <div className="flex-1">
                {/* Video area - now takes full width */}
                <div className="relative h-full bg-black flex items-center justify-center">
                    {isConnecting ? (
                        <div className="text-center text-white animate-pulse">
                            <div className="relative w-16 h-16 mx-auto mb-6">
                                <div className="absolute inset-0 border-4 border-t-primary border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"></div>
                                <div className="absolute inset-2 border-4 border-t-transparent border-r-primary border-b-transparent border-l-transparent rounded-full animate-spin reverse"></div>
                            </div>
                            <p className="text-2xl font-medium">
                                Đang kết nối {callWith === "expert" ? "với chuyên gia" : "với Ami"}...
                            </p>
                        </div>
                    ) : (
                        <div className="w-full h-full relative">
                            {/* Remote video/avatar */}
                            {callWith === "ami" ? (
                                <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-primary/10 to-background">
                                    <AmiAvatar size="xl" mood="happy" className="scale-125 transition-transform duration-500 animate-bounce" />
                                </div>
                            ) : (
                                <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
                                    {isCameraOn ? (
                                        <video
                                            ref={remoteVideoRef}
                                            autoPlay
                                            playsInline
                                            className="w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="text-center text-white">
                                            <div className="w-40 h-40 rounded-full bg-primary/20 flex items-center justify-center mx-auto border-2 border-primary/30">
                                                <User className="w-20 h-20" />
                                            </div>
                                            <p className="mt-4 text-lg">Camera đã tắt</p>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Local video preview */}
                            <div className="absolute bottom-6 right-6 w-40 h-28 bg-gray-900 rounded-xl border-2 border-white/10 overflow-hidden shadow-lg z-20 transition-all duration-300 hover:scale-110">
                                {isCameraOn ? (
                                    <video
                                        ref={localVideoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className="w-full h-full object-cover"
                                        style={{ transform: 'scaleX(-1)' }}
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center bg-gray-800">
                                        <div className="text-white/80 text-sm font-medium">Camera tắt</div>
                                    </div>
                                )}
                            </div>

                            {/* Expert Info Overlay (if applicable) */}
                            {callWith === "expert" && expertInfo && (
                                <div className="absolute top-6 left-6 bg-background/80 backdrop-blur-sm rounded-lg p-4 shadow-lg border border-border">
                                    <p className="text-lg font-semibold">{expertInfo.name}</p>
                                    <p className="text-sm text-muted-foreground">{expertInfo.specialty}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Call controls */}
            <div className="h-24 bg-background/95 backdrop-blur-sm flex items-center justify-center gap-8 border-t border-border shadow-lg">
                {/* Mic Button */}
                {/* <Button
          variant="outline"
          size="icon"
          className={cn(
            "rounded-full w-16 h-16 border-2 shadow-md transition-all duration-200 hover:scale-105",
            !isMicOn && "bg-destructive/20 text-destructive border-destructive/50"
          )}
          onClick={handleToggleMic}
          title={isMicOn ? "Tắt mic" : "Bật mic"}
        >
          {isMicOn ?
            <Mic className="h-7 w-7" /> :
            <MicOff className="h-7 w-7" />
          }
        </Button> */}

                {/* End Call Button */}
                <Button
                    variant="destructive"
                    size="icon"
                    className="rounded-full w-20 h-20 border-2 shadow-lg transition-all duration-200 hover:scale-105 hover:bg-destructive/90"
                    onClick={handleEndCall}
                    title="Kết thúc cuộc gọi"
                >
                    <PhoneOff className="h-8 w-8" />
                </Button>

                {/* Speech Recognition Button (only for Ami calls) */}
                {callWith === "ami" && (
                    <Button
                        variant="outline"
                        size="icon"
                        className={cn(
                            "rounded-full w-16 h-16 border-2 shadow-md transition-all duration-200 hover:scale-105",
                            listening
                                ? "bg-green-500/20 text-green-600 border-green-500/50 animate-pulse"
                                : "bg-blue-500/20 text-blue-600 border-blue-500/30"
                        )}
                        onClick={toggleListening}
                        title={listening ? "Dừng lắng nghe" : "Bắt đầu lắng nghe"}
                    >
                        {listening ? (
                            <div className="relative">
                                <Mic className="h-7 w-7" />
                                <span className="absolute -top-2 -right-2 flex h-4 w-4">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-4 w-4 bg-green-500"></span>
                </span>
                            </div>
                        ) : (
                            <Mic className="h-7 w-7" />
                        )}
                    </Button>
                )}
            </div>
        </div>
    );
};

export default VideoCall;