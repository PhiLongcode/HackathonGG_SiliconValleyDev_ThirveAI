class AIAssistant {
    constructor() {
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.live2dModel = null;
        this.app = null;
        
        this.initializeUI();
        this.setupEventListeners();
        this.initializeLive2D();
        this.setupWebcam();
    }

    initializeUI() {
        this.startButton = document.getElementById('startRecording');
        this.stopButton = document.getElementById('stopRecording');
        this.statusText = document.getElementById('status-text');
        this.conversationHistory = document.getElementById('conversation-history');
        this.live2dCanvas = document.getElementById('live2d');
    }

    setupEventListeners() {
        this.startButton.addEventListener('click', () => this.startRecording());
        this.stopButton.addEventListener('click', () => this.stopRecording());
    }

    async setupWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const video = document.getElementById('webcam');
            video.srcObject = stream;
        } catch (error) {
            console.error('Error accessing webcam:', error);
        }
    }

    async initializeLive2D() {
        try {
            // Initialize PIXI Application
            this.app = new PIXI.Application({
                view: this.live2dCanvas,
                transparent: true,
                autoStart: true,
                width: 800,
                height: 600
            });

            // Load Live2D model
            const model = await PIXI.live2d.Live2DModel.from('/static/models/haru/Haru.model3.json');
            
            // Add model to stage
            this.app.stage.addChild(model);
            this.live2dModel = model;

            // Center the model
            model.x = this.app.screen.width / 2;
            model.y = this.app.screen.height / 2;
            model.scale.set(0.5); // Adjust scale as needed

            // Enable dragging (optional)
            model.interactive = true;
            model.buttonMode = true;
            model
                .on('pointerdown', this.onDragStart)
                .on('pointerup', this.onDragEnd)
                .on('pointerupoutside', this.onDragEnd)
                .on('pointermove', this.onDragMove);

            // Setup idle animation
            this.setupIdleAnimation();
        } catch (error) {
            console.error('Error loading Live2D model:', error);
            this.statusText.textContent = 'Error loading AI avatar';
        }
    }

    setupIdleAnimation() {
        if (!this.live2dModel) return;
        
        // Basic breathing animation
        let t = 0;
        const animate = () => {
            t += 0.05;
            const breathing = Math.sin(t) * 0.1 + 1;
            this.live2dModel.scale.set(breathing);
            requestAnimationFrame(animate);
        };
        animate();
    }

    updateAvatarExpression(sentiment) {
        if (!this.live2dModel) return;
        
        // Update avatar expression based on sentiment
        // This is a placeholder - actual implementation depends on your Live2D model
        switch(sentiment) {
            case 'happy':
                this.live2dModel.expression('happy');
                break;
            case 'sad':
                this.live2dModel.expression('sad');
                break;
            case 'neutral':
                this.live2dModel.expression('neutral');
                break;
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                }
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                await this.processAudio();
            };

            this.mediaRecorder.start(100); // Collect 100ms chunks
            this.isRecording = true;
            this.updateUI(true);
            this.statusText.textContent = 'Recording...';
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.statusText.textContent = 'Error: Could not access microphone';
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateUI(false);
            this.statusText.textContent = 'Processing...';
        }
    }

    updateUI(isRecording) {
        this.startButton.disabled = isRecording;
        this.stopButton.disabled = !isRecording;
    }

    async processAudio() {
        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            
            // Create form data
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');

            // Send to server
            const response = await fetch('/api/process-audio', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Add messages to conversation history
            this.addMessage(data.user_text, 'user');
            this.addMessage(data.ai_text, 'ai');

            // Play AI response audio
            await this.playAudioResponse(data.ai_audio);
            
            this.statusText.textContent = 'Ready to start...';
            
        } catch (error) {
            console.error('Error processing audio:', error);
            this.statusText.textContent = 'Error processing audio';
            this.addMessage('Error: ' + error.message, 'error');
        }
    }

    addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.textContent = text;
        this.conversationHistory.appendChild(messageDiv);
        this.conversationHistory.scrollTop = this.conversationHistory.scrollHeight;
    }

    async playAudioResponse(audioBase64) {
        try {
            // Convert base64 to blob
            const audioData = atob(audioBase64);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const view = new Uint8Array(arrayBuffer);
            for (let i = 0; i < audioData.length; i++) {
                view[i] = audioData.charCodeAt(i);
            }
            const audioBlob = new Blob([arrayBuffer], { type: 'audio/mp3' });
            const audioUrl = URL.createObjectURL(audioBlob);
            
            const audio = new Audio(audioUrl);
            
            // Sync audio with avatar lip movement
            audio.addEventListener('play', () => {
                this.startLipSync();
            });
            
            audio.addEventListener('ended', () => {
                this.stopLipSync();
                URL.revokeObjectURL(audioUrl);
            });
            
            await audio.play();
        } catch (error) {
            console.error('Error playing audio response:', error);
            this.addMessage('Error playing audio response', 'error');
        }
    }

    startLipSync() {
        if (!this.live2dModel) return;
        
        // Animate mouth movement
        this.lipSyncInterval = setInterval(() => {
            const mouthOpenness = Math.random() * 0.8; // Random mouth movement
            this.live2dModel.setParameterValueById('ParamMouthOpenY', mouthOpenness);
        }, 100);
    }

    stopLipSync() {
        if (this.lipSyncInterval) {
            clearInterval(this.lipSyncInterval);
            if (this.live2dModel) {
                this.live2dModel.setParameterValueById('ParamMouthOpenY', 0);
            }
        }
    }
}

// Initialize the application when the page loads
window.addEventListener('DOMContentLoaded', () => {
    new AIAssistant();
}); 