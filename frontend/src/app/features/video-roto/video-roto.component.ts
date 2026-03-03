import { Component, inject, signal, viewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { BoxPrompt } from '../../core/models/api.models';

@Component({
  selector: 'app-video-roto',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './video-roto.component.html',
  styleUrl: './video-roto.component.scss'
})
export class VideoRotoComponent {
  private readonly api = inject(ApiService);

  // File and video state
  selectedFile = signal<File | null>(null);
  videoUrl = signal<string | null>(null);

  // Frame capture state
  isSelectMode = signal(false);
  capturedFrame = signal<string | null>(null);

  // Selection mode: 'point' or 'box'
  selectionMode = signal<'point' | 'box'>('point');

  // Click position (point mode)
  clickX = signal<number | null>(null);
  clickY = signal<number | null>(null);
  currentFrame = signal(0);

  // Bounding box state (box mode)
  boxStart = signal<{ x: number; y: number } | null>(null);
  boxEnd = signal<{ x: number; y: number } | null>(null);
  isDrawingBox = signal(false);

  // Form inputs
  objectLabel = signal('object');
  outputFormat = signal<'silhouette' | 'nuke' | 'exr'>('silhouette');

  // Processing state
  isProcessing = signal(false);
  progress = signal(0);
  error = signal<string | null>(null);
  result = signal<{ url: string; filename: string } | null>(null);

  // Mask preview state
  isLoadingPreview = signal(false);
  maskPreviewUrl = signal<string | null>(null);

  // Playback state
  isPlaying = signal(false);

  // Element references
  readonly videoElement = viewChild<ElementRef<HTMLVideoElement>>('videoPlayer');
  readonly frameCanvas = viewChild<ElementRef<HTMLCanvasElement>>('frameCanvas');
  readonly overlayCanvas = viewChild<ElementRef<HTMLCanvasElement>>('overlayCanvas');

  private videoWidth = 0;
  private videoHeight = 0;
  private videoDuration = 0;
  private videoFps = 24; // Default, will try to detect

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      const file = input.files[0];
      this.selectedFile.set(file);
      this.resetState();

      // Create video URL for preview
      const url = URL.createObjectURL(file);
      this.videoUrl.set(url);
    }
  }

  onVideoLoaded(): void {
    const video = this.videoElement()?.nativeElement;
    if (video) {
      this.videoWidth = video.videoWidth;
      this.videoHeight = video.videoHeight;
      this.videoDuration = video.duration;

      // Debug: Log video dimensions
      console.log('Video Loaded:', {
        videoWidth: this.videoWidth,
        videoHeight: this.videoHeight,
        duration: this.videoDuration
      });

      // Validate video loaded correctly
      if (this.videoWidth === 0 || this.videoHeight === 0) {
        this.error.set('Video dimensions could not be read. The format may not be supported.');
        return;
      }

      // Try to estimate FPS from video metadata (not always available)
      // Default to 24fps for film/vfx work
      this.videoFps = 24;

      // Clear any previous error
      this.error.set(null);

      // Listen for play/pause events
      video.addEventListener('play', () => this.isPlaying.set(true));
      video.addEventListener('pause', () => this.isPlaying.set(false));
      video.addEventListener('error', () => {
        this.error.set('Video format not supported. Try converting to MP4 (H.264) format.');
      });
    }
  }

  onVideoError(): void {
    this.error.set('Could not load video. The format may not be supported by your browser. Try MP4 (H.264) or WebM format.');
  }

  togglePlay(): void {
    const video = this.videoElement()?.nativeElement;
    if (!video || !this.isVideoReady()) return;

    if (video.paused) {
      video.play().catch(err => {
        console.error('Play failed:', err);
        this.error.set('Could not play video. The format may not be supported by your browser.');
      });
    } else {
      video.pause();
    }
  }

  stepForward(): void {
    const video = this.videoElement()?.nativeElement;
    if (!video || !this.isVideoReady()) return;

    video.pause();
    const newTime = video.currentTime + (1 / this.videoFps);
    if (isFinite(newTime) && isFinite(video.duration)) {
      video.currentTime = Math.min(newTime, video.duration);
    }
  }

  stepBackward(): void {
    const video = this.videoElement()?.nativeElement;
    if (!video || !this.isVideoReady()) return;

    video.pause();
    const newTime = video.currentTime - (1 / this.videoFps);
    if (isFinite(newTime)) {
      video.currentTime = Math.max(newTime, 0);
    }
  }

  private isVideoReady(): boolean {
    const video = this.videoElement()?.nativeElement;
    return video !== undefined &&
           video.readyState >= 2 &&
           isFinite(video.duration) &&
           video.duration > 0;
  }

  captureCurrentFrame(): void {
    const video = this.videoElement()?.nativeElement;

    if (!video) return;

    // Create a temporary canvas to capture the frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = this.videoWidth;
    tempCanvas.height = this.videoHeight;

    const ctx = tempCanvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, this.videoWidth, this.videoHeight);

    // Store current frame number
    this.currentFrame.set(Math.floor(video.currentTime * this.videoFps));

    // Switch to select mode and store captured frame as data URL
    this.capturedFrame.set(tempCanvas.toDataURL('image/jpeg', 0.95));
    this.isSelectMode.set(true);

    // After view updates, set up the frame canvas
    setTimeout(() => this.setupFrameCanvas(), 50);
  }

  private setupFrameCanvas(): void {
    const frameCanvas = this.frameCanvas()?.nativeElement;
    if (!frameCanvas || !this.capturedFrame()) return;

    // Load the captured image onto the canvas
    const img = new Image();
    img.onload = () => {
      // Calculate display size (max 900px wide, maintain aspect ratio)
      const maxWidth = 900;
      const maxHeight = 500;
      let displayWidth = this.videoWidth;
      let displayHeight = this.videoHeight;

      if (displayWidth > maxWidth) {
        displayHeight = (maxWidth / displayWidth) * displayHeight;
        displayWidth = maxWidth;
      }
      if (displayHeight > maxHeight) {
        displayWidth = (maxHeight / displayHeight) * displayWidth;
        displayHeight = maxHeight;
      }

      frameCanvas.width = displayWidth;
      frameCanvas.height = displayHeight;
      frameCanvas.style.width = `${displayWidth}px`;
      frameCanvas.style.height = `${displayHeight}px`;

      const ctx = frameCanvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
      }
    };
    img.src = this.capturedFrame()!;
  }

  // Convert display coordinates to video coordinates
  private displayToVideoCoords(displayX: number, displayY: number, rect: DOMRect): { x: number; y: number } {
    const scaleX = this.videoWidth / rect.width;
    const scaleY = this.videoHeight / rect.height;
    const x = Math.round(displayX * scaleX);
    const y = Math.round(displayY * scaleY);
    return {
      x: Math.max(0, Math.min(x, this.videoWidth - 1)),
      y: Math.max(0, Math.min(y, this.videoHeight - 1))
    };
  }

  onFrameClick(event: MouseEvent): void {
    // In box mode, clicks are handled by mousedown/up
    if (this.selectionMode() === 'box') return;

    const canvas = this.frameCanvas()?.nativeElement;
    if (!canvas) return;

    // Validate video dimensions are available
    if (this.videoWidth === 0 || this.videoHeight === 0) {
      this.error.set('Video dimensions not available. Please reload the video.');
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    const videoCoords = this.displayToVideoCoords(clickX, clickY, rect);

    console.log('Point Click:', { displayX: clickX, displayY: clickY, videoX: videoCoords.x, videoY: videoCoords.y });

    this.clickX.set(videoCoords.x);
    this.clickY.set(videoCoords.y);

    this.drawClickMarker(clickX, clickY, videoCoords.x, videoCoords.y);

    // Get mask preview with point
    this.getSegmentationPreview(videoCoords.x, videoCoords.y);
  }

  onFrameMouseDown(event: MouseEvent): void {
    if (this.selectionMode() !== 'box') return;

    const canvas = this.frameCanvas()?.nativeElement;
    if (!canvas) return;

    if (this.videoWidth === 0 || this.videoHeight === 0) {
      this.error.set('Video dimensions not available. Please reload the video.');
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;
    const videoCoords = this.displayToVideoCoords(displayX, displayY, rect);

    this.isDrawingBox.set(true);
    this.boxStart.set(videoCoords);
    this.boxEnd.set(videoCoords);

    // Clear previous selection
    this.clearOverlay();
  }

  onFrameMouseMove(event: MouseEvent): void {
    if (!this.isDrawingBox() || this.selectionMode() !== 'box') return;

    const canvas = this.frameCanvas()?.nativeElement;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;
    const videoCoords = this.displayToVideoCoords(displayX, displayY, rect);

    this.boxEnd.set(videoCoords);

    // Draw the box preview
    this.drawBoxPreview();
  }

  onFrameMouseUp(event: MouseEvent): void {
    if (!this.isDrawingBox() || this.selectionMode() !== 'box') return;

    const canvas = this.frameCanvas()?.nativeElement;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;
    const videoCoords = this.displayToVideoCoords(displayX, displayY, rect);

    this.boxEnd.set(videoCoords);
    this.isDrawingBox.set(false);

    const start = this.boxStart();
    const end = this.boxEnd();

    if (start && end) {
      // Ensure box has minimum size (at least 10px in video coords)
      const boxWidth = Math.abs(end.x - start.x);
      const boxHeight = Math.abs(end.y - start.y);

      if (boxWidth < 10 || boxHeight < 10) {
        console.log('Box too small, ignoring');
        this.clearSelection();
        return;
      }

      console.log('Box Selection:', {
        start,
        end,
        width: boxWidth,
        height: boxHeight
      });

      // Draw final box
      this.drawBoxPreview();

      // Get mask preview with box
      this.getSegmentationPreviewWithBox();
    }
  }

  onFrameMouseLeave(): void {
    // If drawing was in progress, cancel it
    if (this.isDrawingBox()) {
      this.isDrawingBox.set(false);
      this.boxStart.set(null);
      this.boxEnd.set(null);
      this.clearOverlay();
    }
  }

  private drawBoxPreview(): void {
    const frameCanvas = this.frameCanvas()?.nativeElement;
    const overlayCanvas = this.overlayCanvas()?.nativeElement;
    const start = this.boxStart();
    const end = this.boxEnd();

    if (!frameCanvas || !overlayCanvas || !start || !end) return;

    // Match overlay size to frame canvas
    overlayCanvas.width = frameCanvas.width;
    overlayCanvas.height = frameCanvas.height;
    overlayCanvas.style.width = `${frameCanvas.width}px`;
    overlayCanvas.style.height = `${frameCanvas.height}px`;

    const ctx = overlayCanvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Convert video coords to display coords
    const scaleX = overlayCanvas.width / this.videoWidth;
    const scaleY = overlayCanvas.height / this.videoHeight;

    const x1 = Math.min(start.x, end.x) * scaleX;
    const y1 = Math.min(start.y, end.y) * scaleY;
    const x2 = Math.max(start.x, end.x) * scaleX;
    const y2 = Math.max(start.y, end.y) * scaleY;
    const width = x2 - x1;
    const height = y2 - y1;

    // Draw semi-transparent fill
    ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
    ctx.fillRect(x1, y1, width, height);

    // Draw dashed border
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x1, y1, width, height);

    // Draw corner handles
    ctx.setLineDash([]);
    ctx.fillStyle = '#3b82f6';
    const handleSize = 8;
    ctx.fillRect(x1 - handleSize / 2, y1 - handleSize / 2, handleSize, handleSize);
    ctx.fillRect(x2 - handleSize / 2, y1 - handleSize / 2, handleSize, handleSize);
    ctx.fillRect(x1 - handleSize / 2, y2 - handleSize / 2, handleSize, handleSize);
    ctx.fillRect(x2 - handleSize / 2, y2 - handleSize / 2, handleSize, handleSize);

    // Draw coordinate labels
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    const topLabel = `(${Math.min(start.x, end.x)}, ${Math.min(start.y, end.y)})`;
    const bottomLabel = `(${Math.max(start.x, end.x)}, ${Math.max(start.y, end.y)})`;
    const topLabelWidth = ctx.measureText(topLabel).width;
    const bottomLabelWidth = ctx.measureText(bottomLabel).width;

    ctx.fillRect(x1, y1 - 18, topLabelWidth + 6, 16);
    ctx.fillRect(x2 - bottomLabelWidth - 6, y2 + 4, bottomLabelWidth + 6, 16);

    ctx.fillStyle = '#3b82f6';
    ctx.fillText(topLabel, x1 + 3, y1 - 5);
    ctx.fillText(bottomLabel, x2 - bottomLabelWidth - 3, y2 + 16);
  }

  private async getSegmentationPreviewWithBox(): Promise<void> {
    const capturedFrame = this.capturedFrame();
    const start = this.boxStart();
    const end = this.boxEnd();

    if (!capturedFrame || !start || !end) return;

    this.isLoadingPreview.set(true);
    this.maskPreviewUrl.set(null);

    try {
      const response = await fetch(capturedFrame);
      const blob = await response.blob();
      const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });

      // Create box prompt in API contract shape
      const box: BoxPrompt = {
        x1: Math.min(start.x, end.x),
        y1: Math.min(start.y, end.y),
        x2: Math.max(start.x, end.x),
        y2: Math.max(start.y, end.y)
      };

      console.log('Requesting segmentation with box:', box);

      const maskBlob = await this.api.segmentImage(file, undefined, box).toPromise();

      if (maskBlob) {
        const maskUrl = URL.createObjectURL(maskBlob);
        this.maskPreviewUrl.set(maskUrl);
        this.drawMaskOverlayWithBox(maskUrl);
      }
    } catch (err) {
      console.error('Box preview failed:', err);
    } finally {
      this.isLoadingPreview.set(false);
    }
  }

  private drawMaskOverlayWithBox(maskUrl: string): void {
    const frameCanvas = this.frameCanvas()?.nativeElement;
    const overlayCanvas = this.overlayCanvas()?.nativeElement;

    if (!frameCanvas || !overlayCanvas) return;

    const maskImg = new Image();
    maskImg.onload = () => {
      const ctx = overlayCanvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // Create temp canvas for mask processing
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = overlayCanvas.width;
      tempCanvas.height = overlayCanvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return;

      tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);
      const maskData = tempCtx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const data = maskData.data;

      // Create green overlay for mask
      for (let i = 0; i < data.length; i += 4) {
        const maskValue = data[i];
        if (maskValue > 128) {
          data[i] = 34;
          data[i + 1] = 197;
          data[i + 2] = 94;
          data[i + 3] = 100;
        } else {
          data[i + 3] = 0;
        }
      }

      ctx.putImageData(maskData, 0, 0);
      this.drawMaskBorder(ctx, maskData);

      // Redraw box on top
      this.drawBoxPreview();
    };
    maskImg.src = maskUrl;
  }

  private clearOverlay(): void {
    const overlayCanvas = this.overlayCanvas()?.nativeElement;
    if (overlayCanvas) {
      const ctx = overlayCanvas.getContext('2d');
      ctx?.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }

  private async getSegmentationPreview(x: number, y: number): Promise<void> {
    const capturedFrame = this.capturedFrame();
    if (!capturedFrame) return;

    this.isLoadingPreview.set(true);
    this.maskPreviewUrl.set(null);

    try {
      // Convert data URL to File
      const response = await fetch(capturedFrame);
      const blob = await response.blob();
      const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });

      // Call segment-image API
      const maskBlob = await this.api.segmentImage(
        file,
        [{ x, y, label: 1 }]  // foreground point
      ).toPromise();

      if (maskBlob) {
        const maskUrl = URL.createObjectURL(maskBlob);
        this.maskPreviewUrl.set(maskUrl);

        // Draw mask overlay
        this.drawMaskOverlay(maskUrl);
      }
    } catch (err) {
      console.error('Preview failed:', err);
      // Don't show error - preview is optional
    } finally {
      this.isLoadingPreview.set(false);
    }
  }

  private drawMaskOverlay(maskUrl: string): void {
    const frameCanvas = this.frameCanvas()?.nativeElement;
    const overlayCanvas = this.overlayCanvas()?.nativeElement;

    if (!frameCanvas || !overlayCanvas) return;

    const maskImg = new Image();
    maskImg.onload = () => {
      const ctx = overlayCanvas.getContext('2d');
      if (!ctx) return;

      // Clear previous overlay
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

      // Create a temporary canvas for mask processing
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = overlayCanvas.width;
      tempCanvas.height = overlayCanvas.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return;

      // Draw mask scaled to overlay size
      tempCtx.drawImage(maskImg, 0, 0, overlayCanvas.width, overlayCanvas.height);

      // Get mask pixel data
      const maskData = tempCtx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height);
      const data = maskData.data;

      // Create colored overlay (green with transparency)
      for (let i = 0; i < data.length; i += 4) {
        const maskValue = data[i]; // Grayscale value from mask

        if (maskValue > 128) {
          // Object area - green overlay
          data[i] = 34;      // R
          data[i + 1] = 197; // G
          data[i + 2] = 94;  // B
          data[i + 3] = 100; // A (semi-transparent)
        } else {
          // Background - fully transparent
          data[i + 3] = 0;
        }
      }

      // Draw colored mask onto overlay canvas
      ctx.putImageData(maskData, 0, 0);

      // Draw border around mask
      this.drawMaskBorder(ctx, maskData);

      // Redraw click marker on top
      const clickX = this.clickX();
      const clickY = this.clickY();
      if (clickX !== null && clickY !== null) {
        const rect = frameCanvas.getBoundingClientRect();
        const displayX = (clickX / this.videoWidth) * overlayCanvas.width;
        const displayY = (clickY / this.videoHeight) * overlayCanvas.height;
        this.drawClickMarkerOnly(ctx, displayX, displayY, clickX, clickY);
      }
    };
    maskImg.src = maskUrl;
  }

  private drawMaskBorder(ctx: CanvasRenderingContext2D, maskData: ImageData): void {
    const width = maskData.width;
    const height = maskData.height;
    const data = maskData.data;

    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.beginPath();

    // Simple edge detection - draw points where mask changes
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = (y * width + x) * 4;
        const current = data[idx + 3] > 50; // Check alpha

        // Check neighbors
        const left = data[((y) * width + (x - 1)) * 4 + 3] > 50;
        const right = data[((y) * width + (x + 1)) * 4 + 3] > 50;
        const top = data[((y - 1) * width + x) * 4 + 3] > 50;
        const bottom = data[((y + 1) * width + x) * 4 + 3] > 50;

        // If this is an edge pixel (mask pixel with non-mask neighbor)
        if (current && (!left || !right || !top || !bottom)) {
          ctx.fillStyle = '#16a34a';
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }
  }

  private drawClickMarkerOnly(
    ctx: CanvasRenderingContext2D,
    displayX: number,
    displayY: number,
    videoX: number,
    videoY: number
  ): void {
    // Draw crosshair
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;

    ctx.beginPath();
    ctx.moveTo(displayX, displayY - 20);
    ctx.lineTo(displayX, displayY + 20);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(displayX - 20, displayY);
    ctx.lineTo(displayX + 20, displayY);
    ctx.stroke();

    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(displayX, displayY - 20);
    ctx.lineTo(displayX, displayY + 20);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(displayX - 20, displayY);
    ctx.lineTo(displayX + 20, displayY);
    ctx.stroke();

    // Center dot
    ctx.beginPath();
    ctx.arc(displayX, displayY, 6, 0, 2 * Math.PI);
    ctx.fillStyle = '#22c55e';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Coordinate label
    const label = `(${videoX}, ${videoY})`;
    ctx.font = 'bold 12px monospace';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(displayX + 12, displayY - 20, textWidth + 8, 18);
    ctx.fillStyle = '#22c55e';
    ctx.fillText(label, displayX + 16, displayY - 6);
  }

  private drawClickMarker(displayX: number, displayY: number, videoX: number, videoY: number): void {
    const frameCanvas = this.frameCanvas()?.nativeElement;
    const overlayCanvas = this.overlayCanvas()?.nativeElement;

    if (!frameCanvas || !overlayCanvas) return;

    // Match overlay size to frame canvas
    overlayCanvas.width = frameCanvas.width;
    overlayCanvas.height = frameCanvas.height;
    overlayCanvas.style.width = `${frameCanvas.width}px`;
    overlayCanvas.style.height = `${frameCanvas.height}px`;

    const ctx = overlayCanvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Draw crosshair
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 3;

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(displayX, displayY - 30);
    ctx.lineTo(displayX, displayY + 30);
    ctx.stroke();

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(displayX - 30, displayY);
    ctx.lineTo(displayX + 30, displayY);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(displayX, displayY, 12, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(34, 197, 94, 0.4)';
    ctx.fill();
    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Draw coordinate label with background
    const label = `(${videoX}, ${videoY})`;
    ctx.font = 'bold 14px monospace';
    const textWidth = ctx.measureText(label).width;

    // Label background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(displayX + 18, displayY - 28, textWidth + 10, 22);

    // Label text
    ctx.fillStyle = '#22c55e';
    ctx.fillText(label, displayX + 23, displayY - 12);
  }

  backToVideo(): void {
    this.isSelectMode.set(false);
    this.capturedFrame.set(null);
    this.clickX.set(null);
    this.clickY.set(null);

    const overlayCanvas = this.overlayCanvas()?.nativeElement;
    if (overlayCanvas) {
      const ctx = overlayCanvas.getContext('2d');
      ctx?.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }

  clearClick(): void {
    this.clearSelection();
  }

  clearSelection(): void {
    this.clickX.set(null);
    this.clickY.set(null);
    this.boxStart.set(null);
    this.boxEnd.set(null);
    this.isDrawingBox.set(false);
    this.maskPreviewUrl.set(null);
    this.clearOverlay();
  }

  setSelectionMode(mode: 'point' | 'box'): void {
    if (this.selectionMode() !== mode) {
      this.selectionMode.set(mode);
      this.clearSelection();
    }
  }

  hasSelection(): boolean {
    if (this.selectionMode() === 'point') {
      return this.clickX() !== null && this.clickY() !== null;
    } else {
      const start = this.boxStart();
      const end = this.boxEnd();
      return start !== null && end !== null && !this.isDrawingBox();
    }
  }

  getBoxCoordinates(): [number, number, number, number] | null {
    const start = this.boxStart();
    const end = this.boxEnd();
    if (!start || !end) return null;
    return [
      Math.min(start.x, end.x),
      Math.min(start.y, end.y),
      Math.max(start.x, end.x),
      Math.max(start.y, end.y)
    ];
  }

  private resetState(): void {
    this.isSelectMode.set(false);
    this.capturedFrame.set(null);
    this.clickX.set(null);
    this.clickY.set(null);
    this.boxStart.set(null);
    this.boxEnd.set(null);
    this.isDrawingBox.set(false);
    this.currentFrame.set(0);
    this.error.set(null);
    this.result.set(null);
    this.progress.set(0);
    this.maskPreviewUrl.set(null);
    this.isLoadingPreview.set(false);
  }

  async processVideo(): Promise<void> {
    const file = this.selectedFile();

    if (!file || !this.hasSelection()) {
      this.error.set('Please select a video and select the object to track (click or draw box)');
      return;
    }

    const isBoxMode = this.selectionMode() === 'box';
    const boxCoords = this.getBoxCoordinates();
    const x = this.clickX();
    const y = this.clickY();

    // Debug: Log what we're sending to the API
    console.log('API Request:', {
      file: file.name,
      mode: this.selectionMode(),
      clickX: x,
      clickY: y,
      box: boxCoords,
      frame: this.currentFrame(),
      videoWidth: this.videoWidth,
      videoHeight: this.videoHeight,
      label: this.objectLabel(),
      format: this.outputFormat()
    });

    this.isProcessing.set(true);
    this.error.set(null);
    this.progress.set(10);

    try {
      const resultBlob = await this.api.quickRoto(
        file,
        isBoxMode ? null : x,
        isBoxMode ? null : y,
        this.currentFrame(),
        this.objectLabel(),
        this.outputFormat(),
        isBoxMode ? boxCoords ?? undefined : undefined
      ).toPromise();

      this.progress.set(100);

      if (resultBlob) {
        let ext: string;
        switch (this.outputFormat()) {
          case 'silhouette':
            ext = '.fxs';
            break;
          case 'exr':
            ext = '_exr.zip';
            break;
          default:
            ext = '.nk';
        }
        const filename = `${this.objectLabel()}${ext}`;
        const url = URL.createObjectURL(resultBlob);

        this.result.set({ url, filename });
      }
    } catch (err: any) {
      this.error.set(err.message || 'Processing failed');
    } finally {
      this.isProcessing.set(false);
    }
  }

  downloadResult(): void {
    const res = this.result();
    if (!res) return;

    const link = document.createElement('a');
    link.href = res.url;
    link.download = res.filename;
    link.click();
  }

  seekToFrame(frame: number): void {
    const video = this.videoElement()?.nativeElement;
    if (video) {
      video.currentTime = frame / this.videoFps;
    }
  }

  getCurrentTime(): string {
    const video = this.videoElement()?.nativeElement;
    if (!video) return '0:00';

    const time = video.currentTime;
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
}
