/**
* @file renderer.h
 * @brief Declarations for rendering functions.
 *
 * This file declares functions for creating textures and drawing particles on the screen.
 */
#ifndef RENDERER_H
#define RENDERER_H

#include <raylib.h>

/**
 * @brief Creates a circle texture for drawing particles.
 *
 * @param radius The radius of the circle in pixels.
 * @param col The color of the circle.
 * @return RenderTexture2D The generated circle texture.
 */
RenderTexture2D CreateCircleTexture(float radius, Color col);

/**
 * @brief Compute a color based on the particle's velocity.
 *
 * @param speed The speed of the particle.
 * @param maxVel Maximum expected velocity for color scaling.
 * @return Color The computed color.
 */
Color getVelocityColor(float speed, float maxVel);

#endif // RENDERER_H
