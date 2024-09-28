document.addEventListener('DOMContentLoaded', function() {
    const startRecordingBtn = document.getElementById('startRecordingBtn');
    const stopRecordingBtn = document.getElementById('stopRecordingBtn');
    const submitQueryBtn = document.getElementById('submitQueryBtn');
    const transcriptionArea = document.getElementById('transcriptionArea');
    const responseDiv = document.getElementById('responseDiv');
    const languageSelect = document.getElementById('languageSelect');
    const speakResponseBtn = document.getElementById('speakResponseBtn');
    const clearTranscriptionBtn = document.getElementById('clearTranscriptionBtn');

    let mediaRecorder;
    let isRecording = false;
    let transcription = "";
    let currentLanguage = "en-US";
    let isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    let recognition;
    if (!isMobileDevice && 'webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = currentLanguage;
    }

    let speechSynthesisUtterance;
    let isSpeaking = false;
    let youtubePlayer;

    function clearTranscription() {
        transcriptionArea.value = "";
        transcription = "";
        responseDiv.innerHTML = "";
        document.getElementById('videoInfoContainer').style.display = 'none';
        if (youtubePlayer) {
            youtubePlayer.destroy();
            youtubePlayer = null;
        }
    }

    clearTranscriptionBtn.onclick = clearTranscription;

    startRecordingBtn.onclick = async () => {
        if (isRecording) return;

        try {
            if (isMobileDevice) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (event) => {
                    sendAudioChunk(event.data);
                };

                mediaRecorder.start(1000);
                setupSSE();
            } else {
                recognition.start();
                recognition.onresult = (event) => {
                    let interimTranscript = '';
                    let finalTranscript = '';

                    for (let i = event.resultIndex; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            finalTranscript += event.results[i][0].transcript;
                        } else {
                            interimTranscript += event.results[i][0].transcript;
                        }
                    }

                    updateTranscriptionArea(finalTranscript, interimTranscript);
                };
            }

            isRecording = true;
            startRecordingBtn.disabled = true;
            stopRecordingBtn.disabled = false;
            transcriptionArea.value = "Listening...";
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check your permissions.');
        }
    };

    stopRecordingBtn.onclick = () => {
        if (!isRecording) return;
        if (isMobileDevice) {
            mediaRecorder.stop();
            eventSource.close();
        } else {
            recognition.stop();
        }
        isRecording = false;
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
    };

    async function sendAudioChunk(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.wav');

        try {
            await fetch('http://localhost:8081/transcribe', {
                method: 'POST',
                body: formData
            });
        } catch (error) {
            console.error('Error during transcription:', error);
        }
    }

    let eventSource;
    function setupSSE() {
        eventSource = new EventSource('http://localhost:8081/transcribe_stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateTranscriptionArea(data.text, '');
        };
        eventSource.onerror = (error) => {
            console.error('SSE Error:', error);
            eventSource.close();
        };
    }

    function updateTranscriptionArea(finalText, interimText) {
        transcription = finalText;
        transcriptionArea.value = transcription + interimText;
    }

    function updateLanguage() {
        if (isRecording) {
            stopRecordingBtn.onclick();
        }
        currentLanguage = languageSelect.value;
        if (recognition) {
            recognition.lang = currentLanguage;
        }
    }

    languageSelect.onchange = updateLanguage;

    submitQueryBtn.onclick = submitQuery;

    function submitQuery() {
        const query = transcription.trim();
        if (query) {
            setLoadingState(true);
            responseDiv.innerHTML = "Processing your query...";
            speakResponseBtn.disabled = true;
            document.getElementById('videoInfoContainer').style.display = 'none';
            if (youtubePlayer) {
                youtubePlayer.destroy();
                youtubePlayer = null;
            }
            fetch('http://localhost:8081/process_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, language: currentLanguage }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                responseDiv.innerHTML = data.response;
                speakResponseBtn.disabled = false;
                setLoadingState(false);
                if (data.video_id) {
                    updateVideoSpan(data.relevant_start, data.relevant_end);
                    updateFrames(data.frame_urls);
                    createOrUpdateYoutubePlayer(data.video_id, data.relevant_start);
                    document.getElementById('videoInfoContainer').style.display = 'block';
                }
            })
            .catch((error) => {
                console.error('Error:', error);
                responseDiv.innerHTML = `An error occurred while processing your query: ${error.message}`;
                speakResponseBtn.disabled = true;
                setLoadingState(false);
            });
        } else {
            alert("Please record a query before submitting.");
        }
    }

    function setLoadingState(isLoading) {
        submitQueryBtn.disabled = isLoading;
        const loader = submitQueryBtn.querySelector('.loader');
        const buttonText = submitQueryBtn.querySelector('span');
        if (isLoading) {
            loader.style.display = 'inline-block';
            buttonText.style.display = 'none';
        } else {
            loader.style.display = 'none';
            buttonText.style.display = 'inline-block';
        }
    }

    speakResponseBtn.onclick = speakResponse;

    function speakResponse() {
        const textToSpeak = responseDiv.textContent;
        if (textToSpeak) {
            if (!isSpeaking) {
                fetch('http://localhost:8081/text_to_speech', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: textToSpeak, language: currentLanguage }),
                })
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const audio = new Audio(url);
    
                    audio.onended = () => {
                        isSpeaking = false;
                        speakResponseBtn.classList.remove('speaking');
                    };
    
                    audio.play();
                    isSpeaking = true;
                    speakResponseBtn.classList.add('speaking');
                })
                .catch((error) => {
                    console.error('Error:', error);
                    alert('An error occurred while converting text to speech.');
                });
            } else {
                const audioElement = document.querySelector('audio');
                if (audioElement) {
                    audioElement.pause();
                    audioElement.currentTime = 0;
                }
                isSpeaking = false;
                speakResponseBtn.classList.remove('speaking');
            }
        }
    }

    function updateVideoSpan(start, end) {
        const videoSpan = document.getElementById('videoSpan');
        const totalDuration = 600; // Assume 10 minutes total duration
        const startPercent = (start / totalDuration) * 100;
        const endPercent = (end / totalDuration) * 100;
        
        videoSpan.style.setProperty('--start', `${startPercent}%`);
        videoSpan.style.setProperty('--end', `${endPercent}%`);
    }

    function updateFrames(frameUrls) {
        const firstFramesContainer = document.getElementById('firstFrames');
        const lastFramesContainer = document.getElementById('lastFrames');

        firstFramesContainer.innerHTML = '';
        lastFramesContainer.innerHTML = '';

        frameUrls.first_frames.forEach(url => {
            const frameElement = document.createElement('div');
            frameElement.className = 'frame';
            frameElement.style.backgroundImage = `url(${url})`;
            firstFramesContainer.appendChild(frameElement);
        });

        frameUrls.last_frames.forEach(url => {
            const frameElement = document.createElement('div');
            frameElement.className = 'frame';
            frameElement.style.backgroundImage = `url(${url})`;
            lastFramesContainer.appendChild(frameElement);
        });
    }

    function createOrUpdateYoutubePlayer(videoId, startTime) {
        const playerContainer = document.getElementById('youtubePlayer');
        if (youtubePlayer) {
            youtubePlayer.loadVideoById({
                videoId: videoId,
                startSeconds: startTime
            });
        } else {
            youtubePlayer = new YT.Player('youtubePlayer', {
                height: '360',
                width: '640',
                videoId: videoId,
                playerVars: {
                    start: Math.floor(startTime),
                    autoplay: 1
                },
                events: {
                    'onReady': onPlayerReady
                }
            });
        }
    }

    function onPlayerReady(event) {
        event.target.playVideo();
    }
});
