import { Component, inject, signal, viewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { PointPrompt } from '../../core/models/api.models';

@Component({
  selector: 'app-segmentation',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './segmentation.component.html',
  styleUrl: './segmentation.component.scss'
})
export class SegmentationComponent {
  private readonly api = inject(ApiService);

  // Signals for reactive state
  selectedFile = signal<File | null>(null);
  imagePreview = signal<string | null>(null);
  maskPreview = signal<string | null>(null);
  points = signal<PointPrompt[]>([]);
  isProcessing = signal(false);
  error = signal<string | null>(null);
  maskScore = signal<number | null>(null);

  // Canvas for click handling
  readonly canvas = viewChild<ElementRef<HTMLCanvasElement>>('canvas');
  readonly imageElement = viewChild<ElementRef<HTMLImageElement>>('previewImage');

  private imageWidth = 0;
  private imageHeight = 0;

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      const file = input.files[0];
      this.selectedFile.set(file);
      this.points.set([]);
      this.maskPreview.set(null);
      this.error.set(null);

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        this.imagePreview.set(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }

  onImageLoad(event: Event): void {
    const img = event.target as HTMLImageElement;
    this.imageWidth = img.naturalWidth;
    this.imageHeight = img.naturalHeight;
  }

  onCanvasClick(event: MouseEvent): void {
    if (!this.imagePreview()) return;

    const canvas = this.canvas()?.nativeElement;
    const img = this.imageElement()?.nativeElement;
    if (!canvas || !img) return;

    // Get click position relative to image
    const rect = img.getBoundingClientRect();
    const scaleX = this.imageWidth / rect.width;
    const scaleY = this.imageHeight / rect.height;

    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top) * scaleY);

    // Left click = foreground (1), right click would be background (0)
    const label: 0 | 1 = event.button === 2 ? 0 : 1;

    const newPoint: PointPrompt = { x, y, label };
    this.points.update(pts => [...pts, newPoint]);

    this.drawPoints();
  }

  onRightClick(event: MouseEvent): void {
    event.preventDefault();
    if (!this.imagePreview()) return;

    const img = this.imageElement()?.nativeElement;
    if (!img) return;

    const rect = img.getBoundingClientRect();
    const scaleX = this.imageWidth / rect.width;
    const scaleY = this.imageHeight / rect.height;

    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top) * scaleY);

    // Right click = background point
    const newPoint: PointPrompt = { x, y, label: 0 };
    this.points.update(pts => [...pts, newPoint]);

    this.drawPoints();
  }

  private drawPoints(): void {
    const canvas = this.canvas()?.nativeElement;
    const img = this.imageElement()?.nativeElement;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas to image display size
    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    // Clear and draw points
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = rect.width / this.imageWidth;
    const scaleY = rect.height / this.imageHeight;

    for (const point of this.points()) {
      const displayX = point.x * scaleX;
      const displayY = point.y * scaleY;

      ctx.beginPath();
      ctx.arc(displayX, displayY, 8, 0, 2 * Math.PI);
      ctx.fillStyle = point.label === 1 ? '#22c55e' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  clearPoints(): void {
    this.points.set([]);
    this.maskPreview.set(null);
    const canvas = this.canvas()?.nativeElement;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  removeLastPoint(): void {
    this.points.update(pts => pts.slice(0, -1));
    this.drawPoints();
  }

  async segment(): Promise<void> {
    const file = this.selectedFile();
    const pts = this.points();

    if (!file) {
      this.error.set('Please select an image first');
      return;
    }

    if (pts.length === 0) {
      this.error.set('Please click on the image to add at least one point');
      return;
    }

    this.isProcessing.set(true);
    this.error.set(null);

    try {
      const maskBlob = await this.api.segmentImage(file, pts).toPromise();
      if (maskBlob) {
        const maskUrl = URL.createObjectURL(maskBlob);
        this.maskPreview.set(maskUrl);
      }
    } catch (err: any) {
      this.error.set(err.message || 'Segmentation failed');
    } finally {
      this.isProcessing.set(false);
    }
  }

  downloadMask(): void {
    const maskUrl = this.maskPreview();
    if (!maskUrl) return;

    const link = document.createElement('a');
    link.href = maskUrl;
    link.download = 'mask.png';
    link.click();
  }
}
