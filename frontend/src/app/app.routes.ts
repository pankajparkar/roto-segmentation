import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./features/home/home.component').then(m => m.HomeComponent)
  },
  {
    path: 'segment',
    loadComponent: () => import('./features/segmentation/segmentation.component').then(m => m.SegmentationComponent)
  },
  {
    path: 'video-roto',
    loadComponent: () => import('./features/video-roto/video-roto.component').then(m => m.VideoRotoComponent)
  },
  {
    path: '**',
    redirectTo: ''
  }
];
