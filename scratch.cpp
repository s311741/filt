
#if 0
  { // gauss
    int redzone = radius * (width + 1);
    for (int origin = redzone; origin < total_pixels - redzone; ++origin) {
      float value = 0.f;
      float weight = 1.f;
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          int offset = origin + dy * width + dx;
          float gspace = std::exp((dx * dx + dy * dy) * (-1.f / (1 + 2 * radius)));
          weight += gspace;
          value += gspace * s.color[offset];
        }
      }
      s.dst[origin] = value / weight;
    }
    return;
  }
#endif

