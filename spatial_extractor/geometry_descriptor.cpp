// Copyright (c) 2026 Wuhan University, Binglin Hao. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Spatial project: native component-level geometry descriptors.

#include <cmath>
#include <cstdlib>
#include <vector>

// Components must contain strictly more than this many pixels.
// Keep this value near the top so the area filter is easy to tune.
constexpr int kMinComponentArea = 50;
constexpr int kNumClasses = 19;

#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT 512
#endif
#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 512
#endif
constexpr int kImagePixels = IMAGE_HEIGHT * IMAGE_WIDTH;

extern "C" int geometry_descriptor_height() { return IMAGE_HEIGHT; }
extern "C" int geometry_descriptor_width() { return IMAGE_WIDTH; }
extern "C" int geometry_descriptor_min_area() { return kMinComponentArea; }

extern "C" void compute_geometry_descriptor(
    const int *semantic_labels,
    const float *depth_map,
    float *concentration_map,
    float *direction_x_map,
    float *direction_y_map,
    float *direction_z_map);

struct PixelData
{
  float x;
  float y;
  float z;
  int label;
};

struct Vector3
{
  float x;
  float y;
  float z;
};

struct Component
{
  float x;
  float y;
  float z;
  int class_id;
  float concentration;
  float direction_x;
  float direction_y;
  float direction_z;
};

static int find_component_root(int *parent, int x)
{
  int r = x;
  while (parent[r] != r)
  {
    r = parent[r];
  }
  int cur = x;
  while (parent[cur] != r)
  {
    int nxt = parent[cur];
    parent[cur] = r;
    cur = nxt;
  }
  return r;
}

static void merge_components(int *parent, int a, int b)
{
  int ra = find_component_root(parent, a);
  int rb = find_component_root(parent, b);
  if (ra != rb)
  {
    parent[rb] = ra;
  }
}

inline void normalized_direction(const Component &source,
                                 const Component &target,
                                 Vector3 &direction)
{
  // Keep this small operation inline; a zero-length direction stays zero.
  float dx = source.x - target.x;
  float dy = source.y - target.y;
  float dz = source.z - target.z;
  float norm = sqrtf(dx * dx + dy * dy + dz * dz);

  if (norm > 0.0f)
  {
    float inv_norm = 1.0f / norm;
    direction.x = dx * inv_norm;
    direction.y = dy * inv_norm;
    direction.z = dz * inv_norm;
  }
  else
  {
    direction.x = direction.y = direction.z = 0.0f;
  }
}

static void zero_feature_map(float *feature_map)
{
  for (int i = 0; i < kImagePixels; i++)
  {
    feature_map[i] = 0.0f;
  }
}

extern "C" void compute_geometry_descriptor(
    const int *semantic_labels,
    const float *depth_map,
    float *concentration_map,
    float *direction_x_map,
    float *direction_y_map,
    float *direction_z_map)
{
  zero_feature_map(concentration_map);
  zero_feature_map(direction_x_map);
  zero_feature_map(direction_y_map);
  zero_feature_map(direction_z_map);
  PixelData *pixels = (PixelData *)malloc(sizeof(PixelData) * kImagePixels);
  if (!pixels)
  {
    return;
  }

  // Segment same-class pixels into eight-connected components.
  // 1. Initialize the disjoint-set labels.
  int *parent = (int *)malloc(sizeof(int) * (kImagePixels + 1));
  if (!parent)
  {
    free(pixels);
    return;
  }
  for (int i = 0; i <= kImagePixels; i++)
  {
    parent[i] = i;
  }
  int next_label = 1;

  // Initialize every pixel, including ignored labels.
  for (int i = 0; i < kImagePixels; i++)
  {
    pixels[i].x = 0.0;
    pixels[i].y = 0.0;
    pixels[i].z = 0;
    pixels[i].label = 0;
  }

  // 2. Assign provisional labels in row-major order and record equivalences.
  for (int y = 0; y < IMAGE_HEIGHT; y++)
  {
    int y_val = y * IMAGE_WIDTH;
    for (int x = 0; x < IMAGE_WIDTH; x++)
    {
      int pixel_index = y_val + x;
      int class_id = semantic_labels[pixel_index];

      // Only the 19 train IDs are valid; 255 and other IDs are ignored.
      if (class_id < 0 || class_id >= kNumClasses)
        continue;

      // Consider already visited neighbors with the same semantic class.
      int neighbor_labels[4];
      int neighbor_count = 0;

      // Above.
      if (y > 0)
      {
        int neighbor_index = y_val - IMAGE_WIDTH + x;
        if (semantic_labels[neighbor_index] == class_id &&
            pixels[neighbor_index].label > 0)
        {
          neighbor_labels[neighbor_count++] = pixels[neighbor_index].label;
        }
      }
      // Left.
      if (x > 0)
      {
        int neighbor_index = y_val + (x - 1);
        if (semantic_labels[neighbor_index] == class_id &&
            pixels[neighbor_index].label > 0)
        {
          neighbor_labels[neighbor_count++] = pixels[neighbor_index].label;
        }
      }
      // Upper left.
      if (y > 0 && x > 0)
      {
        int neighbor_index = y_val - IMAGE_WIDTH + (x - 1);
        if (semantic_labels[neighbor_index] == class_id &&
            pixels[neighbor_index].label > 0)
        {
          neighbor_labels[neighbor_count++] = pixels[neighbor_index].label;
        }
      }
      // Upper right.
      if (y > 0 && x + 1 < IMAGE_WIDTH)
      {
        int neighbor_index = y_val - IMAGE_WIDTH + (x + 1);
        if (semantic_labels[neighbor_index] == class_id &&
            pixels[neighbor_index].label > 0)
        {
          neighbor_labels[neighbor_count++] = pixels[neighbor_index].label;
        }
      }

      if (neighbor_count == 0)
      {
        // A component starts at this pixel.
        pixels[pixel_index].label = next_label;
        parent[next_label] = next_label;
        next_label += 1;
      }
      else
      {
        // Merge all equivalent labels under the smallest neighbor label.
        int min_label = neighbor_labels[0];
        for (int k = 1; k < neighbor_count; k++)
        {
          if (neighbor_labels[k] < min_label)
          {
            min_label = neighbor_labels[k];
          }
        }
        pixels[pixel_index].label = min_label;
        for (int k = 0; k < neighbor_count; k++)
        {
          merge_components(parent, min_label, neighbor_labels[k]);
        }
      }
    }
  }

  // 3. Compress each provisional label to its root.
  for (int i = 1; i < next_label; i++)
  {
    parent[i] = find_component_root(parent, i);
  }

  // 4. Give each root a compact final component label.
  std::vector<int> label_map(next_label, 0);
  int component_count = 0;
  for (int i = 1; i < next_label; i++)
  {
    int r = parent[i];
    if (r == i)
    {
      component_count++;
      label_map[r] = component_count;
    }
  }

  // 5. Accumulate component coordinates and depth in double precision.
  double *sum_x = (double *)malloc(sizeof(double) * (component_count + 1));
  double *sum_y = (double *)malloc(sizeof(double) * (component_count + 1));
  double *sum_z = (double *)malloc(sizeof(double) * (component_count + 1));
  int *component_area = (int *)malloc(sizeof(int) * (component_count + 1));
  if (!sum_x || !sum_y || !sum_z || !component_area)
  {
    free(pixels);
    free(parent);
    free(sum_x);
    free(sum_y);
    free(sum_z);
    free(component_area);
    return;
  }
  // Zero all accumulator slots, including the ignored-label slot.
  for (int i = 0; i <= component_count; i++)
  {
    sum_x[i] = 0.0;
    sum_y[i] = 0.0;
    sum_z[i] = 0.0;
    component_area[i] = 0;
  }

  // 6. Replace provisional labels and accumulate component statistics.
  for (int y = 0; y < IMAGE_HEIGHT; y++)
  {
    int y_val = y * IMAGE_WIDTH;
    for (int x = 0; x < IMAGE_WIDTH; x++)
    {
      int pixel_index = y_val + x;
      int provisional_label = pixels[pixel_index].label;
      int root = parent[provisional_label];
      int final_label = label_map[root];
      pixels[pixel_index].label = final_label;

      // Preserve the legacy coordinate and depth conventions.
      sum_x[final_label] += (double)x;
      sum_y[final_label] += (double)y;
      sum_z[final_label] += (double)depth_map[pixel_index];
      component_area[final_label] += 1;
    }
  }

  // 7. Compute centroids and retain components above the area threshold.
  float *centroid_x = (float *)malloc(sizeof(float) * (component_count + 1));
  float *centroid_y = (float *)malloc(sizeof(float) * (component_count + 1));
  float *centroid_z = (float *)malloc(sizeof(float) * (component_count + 1));
  int *keep_component = (int *)malloc(sizeof(int) * (component_count + 1));
  if (!centroid_x || !centroid_y || !centroid_z || !keep_component)
  {
    free(pixels);
    free(parent);
    free(sum_x);
    free(sum_y);
    free(sum_z);
    free(component_area);
    free(centroid_x);
    free(centroid_y);
    free(centroid_z);
    free(keep_component);
    return;
  }
  // Label zero represents ignored pixels and must not read uninitialized data.
  centroid_x[0] = 0.0f;
  centroid_y[0] = 0.0f;
  centroid_z[0] = 0.0f;
  keep_component[0] = 0;
  for (int component_label = 1; component_label <= component_count;
       component_label++)
  {
    double inverse_area = 1.0 / (double)component_area[component_label];
    centroid_x[component_label] =
        (float)(sum_x[component_label] * inverse_area);
    centroid_y[component_label] =
        (float)(sum_y[component_label] * inverse_area);
    centroid_z[component_label] =
        (float)(sum_z[component_label] * inverse_area);
    keep_component[component_label] =
        component_area[component_label] > kMinComponentArea ? 1 : 0;
  }

  // 8. Broadcast each component centroid back to its pixels.
  for (int i = 0; i < kImagePixels; i++)
  {
    int component_label = pixels[i].label;
    pixels[i].x = centroid_x[component_label];
    pixels[i].y = centroid_y[component_label];
    pixels[i].z = centroid_z[component_label];
  }

  // 9. Release temporary segmentation buffers.
  free(parent);
  free(sum_x);
  free(sum_y);
  free(sum_z);
  free(component_area);
  free(centroid_x);
  free(centroid_y);
  free(centroid_z);

  // Compute the four relation descriptors from component centroids.
  int *label_index = (int *)malloc(sizeof(int) * kImagePixels);
  int *pixel_label_index = (int *)malloc(sizeof(int) * kImagePixels);
  if (!label_index || !pixel_label_index)
  {
    free(pixels);
    free(keep_component);
    free(label_index);
    free(pixel_label_index);
    return;
  }
  for (int i = 0; i < kImagePixels; i++)
  {
    label_index[i] = -1;
    pixel_label_index[i] = -1;
  }
  std::vector<Component> components;
  // Reserve one slot per component to avoid repeated reallocation.
  components.reserve(component_count);
  float class_component_count[kNumClasses] = {0.0f};

  // 1. Build the list of retained components.
  int retained_count = 0;
  for (int i = 0; i < kImagePixels; i++)
  {
    int class_id = semantic_labels[i];
    if (class_id < 0 || class_id >= kNumClasses)
    {
      continue;
    }
    int component_label = pixels[i].label;
    int component_index = label_index[component_label];
    if (keep_component[component_label] > 0 && component_index < 0)
    {
      // First pixel of a retained component.
      Component component;
      component.x = pixels[i].x;
      component.y = pixels[i].y;
      component.z = pixels[i].z;
      component.class_id = class_id;
      component.concentration = 0.0f;
      component.direction_x = 0.0f;
      component.direction_y = 0.0f;
      component.direction_z = 0.0f;
      component_index = retained_count;
      components.push_back(component);
      label_index[component_label] = component_index;
      class_component_count[class_id] += 1.0f;
      retained_count += 1;
    }
    pixel_label_index[i] = component_index;
  }

  if (retained_count == 0)
  {
    // No component survived the pixel-count threshold.
    free(pixels);
    free(keep_component);
    free(label_index);
    free(pixel_label_index);
    return;
  }

  // 2. Accumulate the direction from component i to each semantic class c.
  Vector3 zero_direction = {0.0f, 0.0f, 0.0f};
  std::vector<std::vector<Vector3>> directional(
      retained_count, std::vector<Vector3>(kNumClasses, zero_direction));

  float inverse_class_count[kNumClasses];
  for (int c = 0; c < kNumClasses; c++)
  {
    inverse_class_count[c] = (class_component_count[c] > 0.0f)
                                 ? (1.0f / class_component_count[c])
                                 : 0.0f;
  }

  // 3. Compute pairwise directions, then the four descriptors.
  const float concentration_scale = 1.0f / static_cast<float>(kNumClasses);
  for (int i = 0; i < retained_count; i++)
  {
    const Component &source_component = components[i];
    int source_class = source_component.class_id;
    // Visit each unordered component pair only once.
    for (int j = i + 1; j < retained_count; j++)
    {
      // Self-direction is zero; other same-class pairs are still included.
      const Component &target_component = components[j];
      int target_class = target_component.class_id;
      Vector3 direction;
      normalized_direction(source_component, target_component, direction);
      directional[i][target_class].x += direction.x;
      directional[i][target_class].y += direction.y;
      directional[i][target_class].z += direction.z;

      directional[j][source_class].x -= direction.x;
      directional[j][source_class].y -= direction.y;
      directional[j][source_class].z -= direction.z;
    }

    // Keep the original global aggregation used by the legacy experiments:
    // first sum the class-wise mean direction vectors, then normalize once.
    // The class count includes the current component, as in old experiments.
    Vector3 total_feature = zero_direction;
    for (int c = 0; c < kNumClasses; c++)
    {
      if (class_component_count[c] > 0.0f)
      {
        total_feature.x += directional[i][c].x * inverse_class_count[c];
        total_feature.y += directional[i][c].y * inverse_class_count[c];
        total_feature.z += directional[i][c].z * inverse_class_count[c];
      }
    }

    const float relation_magnitude =
        sqrtf(total_feature.x * total_feature.x +
              total_feature.y * total_feature.y +
              total_feature.z * total_feature.z);
    components[i].concentration = relation_magnitude * concentration_scale;

    // Preserve legacy outputs for non-zero vectors while preventing NaN.
    if (relation_magnitude > 1e-12f)
    {
      const float inverse_magnitude = 1.0f / relation_magnitude;
      components[i].direction_x = total_feature.x * inverse_magnitude;
      components[i].direction_y = total_feature.y * inverse_magnitude;
      components[i].direction_z = total_feature.z * inverse_magnitude;
    }
  }

  // 4. Broadcast component descriptors to their pixels.
  for (int i = 0; i < kImagePixels; i++)
  {
    // Pixels outside retained components keep the initialized zero value.
    int component_index = pixel_label_index[i];
    if (component_index >= 0)
    {
      concentration_map[i] = components[component_index].concentration;
      direction_x_map[i] = components[component_index].direction_x;
      direction_y_map[i] = components[component_index].direction_y;
      direction_z_map[i] = components[component_index].direction_z;
    }
  }

  free(pixels);
  free(keep_component);
  free(label_index);
  free(pixel_label_index);
}
