import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

type QualityPreset = 'lossless' | 'high' | 'medium' | 'stream-copy';

@Component({
  selector: 'app-video-converter',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './video-converter.component.html',
  styleUrl: './video-converter.component.scss'
})
export class VideoConverterComponent {
  private readonly api = inject(ApiService);

  selectedFile = signal<File | null>(null);
  quality = signal<QualityPreset>('high');
  isConverting = signal(false);
  progress = signal(0);
  error = signal<string | null>(null);
  isDragOver = signal(false);

  readonly qualityOptions: { value: QualityPreset; label: string; description: string }[] = [
    { value: 'stream-copy', label: 'Stream Copy', description: 'Fastest - no re-encoding (may not always work)' },
    { value: 'lossless', label: 'Lossless', description: 'CRF 0 - mathematically lossless, largest file' },
    { value: 'high', label: 'High Quality', description: 'CRF 17 - visually lossless (recommended)' },
    { value: 'medium', label: 'Medium', description: 'CRF 23 - good quality, smaller file' },
  ];

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.setFile(input.files[0]);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver.set(false);
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver.set(false);

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      this.setFile(event.dataTransfer.files[0]);
    }
  }

  private setFile(file: File): void {
    const name = file.name.toLowerCase();
    if (!name.endsWith('.mov')) {
      this.error.set('Please select a MOV file');
      return;
    }
    this.selectedFile.set(file);
    this.error.set(null);
  }

  clearFile(): void {
    this.selectedFile.set(null);
    this.error.set(null);
  }

  formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  async convert(): Promise<void> {
    const file = this.selectedFile();
    if (!file) return;

    this.isConverting.set(true);
    this.error.set(null);
    this.progress.set(10);

    try {
      const quality = this.quality();
      let blob: Blob;

      this.progress.set(30);

      if (quality === 'stream-copy') {
        blob = await this.api.streamCopyToMp4(file).toPromise() as Blob;
      } else {
        blob = await this.api.convertMovToMp4(file, quality).toPromise() as Blob;
      }

      this.progress.set(90);

      // Create download link and trigger download
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name.replace(/\.mov$/i, '.mp4');
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      this.progress.set(100);

      // Reset after successful conversion
      setTimeout(() => {
        this.selectedFile.set(null);
        this.progress.set(0);
      }, 1500);

    } catch (err: any) {
      console.error('Conversion failed:', err);
      this.error.set(err.message || 'Conversion failed. Please try a different quality setting.');
    } finally {
      this.isConverting.set(false);
    }
  }
}
