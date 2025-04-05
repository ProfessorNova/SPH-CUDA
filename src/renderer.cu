/**
 * @file renderer.cu
 * @brief Implementation of rendering functions.
 *
 * This file implements functions for creating textures and computing colors for rendering particles.
 */

#include "renderer.h"
#include <raylib.h>

// Implementation of CreateCircleTexture
RenderTexture2D CreateCircleTexture(float radius, Color col) {
    int diameter = static_cast<int>(radius * 2);
    RenderTexture2D texture = LoadRenderTexture(diameter, diameter);
    BeginTextureMode(texture);
    ClearBackground(BLANK);
    DrawCircle(diameter / 2, diameter / 2, radius, col);
    EndTextureMode();
    return texture;
}

// Implementation of getVelocityColor
Color getVelocityColor(float speed, float maxVel) {
    float t = speed / maxVel;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    if (t < 0.33f) {
        float localT = t / 0.33f;
        // Interpolate between black and blue
        constexpr Color c1 = {0, 0, 64, 255};
        constexpr Color c2 = {0, 0, 255, 255};
        Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    }
    if (t < 0.66f) {
        float localT = (t - 0.33f) / 0.33f;
        constexpr Color c1 = {0, 0, 255, 255};
        constexpr Color c2 = {0, 255, 255, 255};
        Color result;
        result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
        result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
        result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
        result.a = 255;
        return result;
    }
    float localT = (t - 0.66f) / 0.34f;
    constexpr Color c1 = {0, 255, 255, 255};
    constexpr Color c2 = {255, 255, 255, 255};
    Color result;
    result.r = static_cast<unsigned char>(c1.r + (c2.r - c1.r) * localT);
    result.g = static_cast<unsigned char>(c1.g + (c2.g - c1.g) * localT);
    result.b = static_cast<unsigned char>(c1.b + (c2.b - c1.b) * localT);
    result.a = 255;
    return result;
}
