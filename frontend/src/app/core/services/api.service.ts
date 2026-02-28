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
   * Quick rotoscoping - upload video, click, get FXS
   */
  quickRoto(
    video: File,
    clickX: number,
    clickY: number,
    frameIdx: number = 0,
    label: string = 'object',
    outputFormat: string = 'silhouette'
  ): Observable<Blob> {
    const formData = new FormData();
    formData.append('video', video);
    formData.append('click_x', clickX.toString());
    formData.append('click_y', clickY.toString());
    formData.append('frame_idx', frameIdx.toString());
    formData.append('label', label);
    formData.append('output_format', outputFormat);

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
}
