import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';
import { DeviceInfo } from '../../core/models/api.models';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent implements OnInit {
  private readonly api = inject(ApiService);

  deviceInfo = signal<DeviceInfo | null>(null);
  isConnected = signal(false);
  isLoading = signal(true);

  ngOnInit(): void {
    this.checkConnection();
  }

  async checkConnection(): Promise<void> {
    this.isLoading.set(true);
    try {
      const info = await this.api.getDeviceInfo().toPromise();
      if (info) {
        this.deviceInfo.set(info);
        this.isConnected.set(true);
      }
    } catch {
      this.isConnected.set(false);
    } finally {
      this.isLoading.set(false);
    }
  }
}
