import { Component, inject, signal, viewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';

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

  // Click position
  clickX = signal<number | null>(null);
  clickY = signal<number | null>(null);
  currentFrame = signal(0);

  // Form inputs
  objectLabel = signal('object');
  outputFormat = signal<'silhouette' | 'nuke'>('silhouette');

  // Processing state
  isProcessing = signal(false);
  progress = signal(0);
  error = signal<string | null>(null);
  result = signal<{ url: string; filename: string } | null>(null);

  // Video element reference
  readonly videoElement = viewChild<ElementRef<HTMLVideoElement>>('videoPlayer');
  readonly canvasOverlay = viewChild<ElementRef<HTMLCanvasElement>>('canvasOverlay');

  private videoWidth = 0;
  private videoHeight = 0;

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
    }
  }

  onVideoClick(event: MouseEvent): void {
    const video = this.videoElement()?.nativeElement;
    if (!video) return;

    const rect = video.getBoundingClientRect();
    const scaleX = this.videoWidth / rect.width;
    const scaleY = this.videoHeight / rect.height;

    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top) * scaleY);

    this.clickX.set(x);
    this.clickY.set(y);
    this.currentFrame.set(Math.floor(video.currentTime * 24)); // Assume 24fps

    this.drawClickMarker(event.clientX - rect.left, event.clientY - rect.top);
  }

  private drawClickMarker(displayX: number, displayY: number): void {
    const canvas = this.canvasOverlay()?.nativeElement;
    const video = this.videoElement()?.nativeElement;
    if (!canvas || !video) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw crosshair
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(displayX, displayY - 20);
    ctx.lineTo(displayX, displayY + 20);
    ctx.stroke();

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(displayX - 20, displayY);
    ctx.lineTo(displayX + 20, displayY);
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(displayX, displayY, 8, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(34, 197, 94, 0.3)';
    ctx.fill();
    ctx.stroke();
  }

  clearClick(): void {
    this.clickX.set(null);
    this.clickY.set(null);

    const canvas = this.canvasOverlay()?.nativeElement;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  private resetState(): void {
    this.clickX.set(null);
    this.clickY.set(null);
    this.currentFrame.set(0);
    this.error.set(null);
    this.result.set(null);
    this.progress.set(0);
  }

  async processVideo(): Promise<void> {
    const file = this.selectedFile();
    const x = this.clickX();
    const y = this.clickY();

    if (!file || x === null || y === null) {
      this.error.set('Please select a video and click on the object to track');
      return;
    }

    this.isProcessing.set(true);
    this.error.set(null);
    this.progress.set(10);

    try {
      const resultBlob = await this.api.quickRoto(
        file,
        x,
        y,
        this.currentFrame(),
        this.objectLabel(),
        this.outputFormat()
      ).toPromise();

      this.progress.set(100);

      if (resultBlob) {
        const ext = this.outputFormat() === 'silhouette' ? '.fxs' : '.nk';
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
      video.currentTime = frame / 24; // Assume 24fps
    }
  }
}
