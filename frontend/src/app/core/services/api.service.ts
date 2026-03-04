import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../../environments/environment';
import { DeviceInfo, ApiInfo, PointPrompt, BoxPrompt } from '../models/api.models';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = environment.apiUrl;

  /**
   * Get API info
   */
  getApiInfo(): Observable<ApiInfo> {
    return this.http.get<ApiInfo>(this.baseUrl);
  }

  /**
   * Get device information (MPS/CUDA/CPU)
   */
  getDeviceInfo(): Observable<DeviceInfo> {
    return this.http.get<DeviceInfo>(`${this.baseUrl}/api/v1/segment/device-info`);
  }

  /**
   * Segment an image with point/box prompts
   * Returns the mask as a Blob (PNG image)
   */
  segmentImage(
    image: File,
    points?: PointPrompt[],
    box?: BoxPrompt
  ): Observable<Blob> {
    const formData = new FormData();
    formData.append('image', image);

    if (points && points.length > 0) {
      const pointsArray = points.map(p => [p.x, p.y, p.label]);
      formData.append('points', JSON.stringify(pointsArray));
    }

    if (box) {
      formData.append('box', JSON.stringify([box.x1, box.y1, box.x2, box.y2]));
    }

    return this.http.post(`${this.baseUrl}/api/v1/segment/segment-image`, formData, {
      responseType: 'blob'
    });
  }

  /**
   * Quick rotoscoping - upload video, click or draw box, get FXS
   * Supports both point selection (clickX, clickY) and box selection (box)
   */
  quickRoto(
    video: File,
    clickX: number | null,
    clickY: number | null,
    frameIdx: number = 0,
    label: string = 'object',
    outputFormat: string = 'silhouette',
    box?: [number, number, number, number],
    annotationPoints?: PointPrompt[]
  ): Observable<Blob> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('frame_idx', frameIdx.toString());
    formData.append('label', label);
    formData.append('output_format', outputFormat);

    // Multiple annotation points (new system)
    if (annotationPoints && annotationPoints.length > 0) {
      const pointsArray = annotationPoints.map(p => [p.x, p.y, p.label]);
      formData.append('points', JSON.stringify(pointsArray));
    }
    // Box selection
    else if (box) {
      formData.append('box', JSON.stringify(box));
    }
    // Legacy single point
    else if (clickX !== null && clickY !== null) {
      formData.append('click_x', clickX.toString());
      formData.append('click_y', clickY.toString());
    }

    return this.http.post(`${this.baseUrl}/api/v1/segment/quick-roto`, formData, {
      responseType: 'blob'
    });
  }

  /**
   * Health check
   */
  healthCheck(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.baseUrl}/health`);
  }

  /**
   * Convert MOV to MP4 without quality loss
   * @param video The MOV file to convert
   * @param quality Quality preset: 'lossless', 'high', 'medium'
   * @returns Observable with the converted MP4 as Blob
   */
  convertMovToMp4(
    video: File,
    quality: 'lossless' | 'high' | 'medium' = 'high'
  ): Observable<Blob> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('quality', quality);

    return this.http.post(`${this.baseUrl}/api/v1/convert/mov-to-mp4`, formData, {
      responseType: 'blob'
    });
  }

  /**
   * Stream copy MOV to MP4 (fastest, no re-encoding)
   * @param video The MOV file to convert
   * @returns Observable with the converted MP4 as Blob
   */
  streamCopyToMp4(video: File): Observable<Blob> {
    const formData = new FormData();
    formData.append('video', video);

    return this.http.post(`${this.baseUrl}/api/v1/convert/stream-copy`, formData, {
      responseType: 'blob'
    });
  }
}
