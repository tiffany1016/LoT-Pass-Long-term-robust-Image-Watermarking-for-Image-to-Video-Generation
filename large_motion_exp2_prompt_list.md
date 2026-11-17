# I2V Large-Motion Experiment — Prompt Catalog

## Usage Notes

- Each prompt is paired with the exact source image filename shown below.
- Unless otherwise specified, the camera should remain fixed and the scene should be a continuous shot.
- Exit cases ask the subject to leave the frame in the first half of the video and stay out of view; verify this in the generated frames.
- Verify the actual motion after generation. An I2V model may not obey the direction, framing, or motion constraints.
- `W3-A` uses the clarified **rightward, in-frame** motion wording to avoid ambiguity about running toward the camera.


## Single-Person Motion

### `walk1.png`

**W1-A — Walk (In Frame)**

```text
The man in the suit continues walking to the right across the crosswalk at a steady pace, carrying his briefcase. He remains visible throughout the video. The camera stays completely fixed, with no panning or zooming.
```

**W1-B — Squat**

```text
The man stops walking and slowly squats down at his current position while holding his briefcase, then stands back up. His entire body remains visible. The camera is completely stationary.
```

**W1-C — Run Exit**

```text
The man suddenly runs quickly to the right across the crosswalk, completely exiting the right edge of the frame within the first half of the video. The camera remains fixed on the empty crosswalk after he leaves, without following him.
```

### `walk2.png`

**W2-A — Walk (In Frame)**

```text
The man continues walking steadily to the right along the green wall, maintaining a natural walking motion. He remains visible throughout the video. The camera stays fixed in place.
```

**W2-B — Squat**

```text
The man stops walking and slowly crouches down near the green wall, bending his knees deeply, then stands up again. His shadow changes naturally with his movement. The camera remains completely stationary.
```

**W2-C — Run Exit**

```text
The man accelerates into a fast run toward the right and completely exits the frame within the first half of the video. The camera stays fixed on the green wall and empty pavement after he disappears.
```

### `walk3.png`

**W3-A — Run in Frame**

```text
The man in the blue shirt continues running to the right with natural arm and leg movements, but remains fully visible throughout the video. The camera remains fixed, showing the same outdoor background.
```

**W3-C — Run Exit**

```text
The man in the blue shirt sprints rapidly toward the right and completely leaves the camera frame within the first half of the video. The camera remains fixed on the outdoor scene after he disappears, without tracking his movement.
```

**W3-D — Stationary Subject, Snowy Background**

```text
A stationary camera captures a man wearing a light blue shirt in an outdoor park. The man remains completely still in the same position, maintaining his original appearance and pose. Meanwhile, the surrounding environment gradually transforms into a snowy winter scene. Snow begins to cover the trees, grass, and ground, while the green foliage turns into snow-covered branches. The man remains unchanged throughout the video.
```

**W3-E — Stationary Subject, Nightfall**

```text
A stationary camera captures a man wearing a light blue shirt in an outdoor park. The man remains completely still in the same position, maintaining his original appearance and pose. Meanwhile, the scene gradually transitions from daylight to nighttime. The sky darkens, the trees and surrounding buildings become dim, and the background gradually falls into darkness. The man stays in the same position throughout the video.
```

**W3-F — Subject and Background Motion**

```text
A man wearing a light blue shirt runs forward through an outdoor park, visibly changing his position within the frame. His body moves naturally as he runs, and he does not remain fixed in the center of the screen. At the same time, the camera moves sideways, causing the trees, buildings, and surrounding scenery to shift noticeably across the frame. Both the man and the background show continuous and clearly visible motion throughout the video.
```

**W3-G — Stationary Subject, Moving Background**

```text
A man wearing a light blue shirt remains completely motionless in the exact same position and pose throughout the video. The camera stays fixed, keeping the man perfectly stationary within the frame. Meanwhile, the background trees sway strongly in the wind, with branches and leaves moving continuously and noticeably. The surrounding foliage moves dynamically, while the man's body, clothing, and position remain unchanged.
```

## Other Subjects: Dogs, Cars, and Birds

### `dog1.png`

**D1-A — Run Exit**

```text
The small dog runs quickly toward the right across the grassy field, carrying the red ball in its mouth. The dog completely exits the right edge of the frame within the first half of the video. The camera remains fixed on the empty grass after the dog leaves.
```

**D1-B — Jump in Place**

```text
The small dog stops near its original position and makes two energetic jumps straight upward while holding the red ball in its mouth. It lands in the same area without moving across the field. The camera remains stationary.
```

### `dog2.png`

**D2-A — Run Exit**

```text
The corgi runs rapidly toward the right across the grass with natural leg movements. It completely exits the right side of the frame within the first half of the video. The camera remains fixed on the empty grassy field afterward.
```

**D2-B — Jump in Place**

```text
The corgi stops running and performs two playful vertical jumps in approximately the same location, lifting all four paws off the ground before landing. The dog remains visible throughout the video, and the camera stays completely fixed.
```

### `car1.png`

**C1-A — Drive in Frame**

```text
The yellow sports car drives smoothly along the curved mountain road toward the right, maintaining its original shape and appearance. The car remains visible throughout the video. The camera stays completely stationary.
```

**C1-B — Drive Exit**

```text
The yellow sports car accelerates along the curved road toward the right and completely exits the frame within the first half of the video. The camera remains fixed on the empty mountain road after the car disappears.
```

**C1-C — Snowy Background Transformation**

```text
A stationary camera captures the yellow convertible on the mountain road. The car remains in nearly the same position and appearance, while the surrounding environment gradually transforms into a snowy winter scene. Snow slowly covers the hillside, grass, and road, and the atmosphere becomes cold and overcast. The car stays visible and unchanged as much as possible.
```

**C1-D — Day-to-Night Transition**

```text
A stationary camera captures the yellow convertible on the mountain road. The car remains in nearly the same position, while the whole scene gradually transitions from daytime to nighttime. The sky slowly darkens, the landscape becomes dim, and the road and hills fade into evening shadows. The headlights of the car turn on naturally as the scene becomes dark.
```

**C1-E — Car and Background Motion**

```text
The yellow convertible must visibly move across the frame, starting near the right side of the image and traveling toward the left side, so that its position in the frame changes clearly over time. The car should not remain fixed in the same place. Meanwhile, the background should also change continuously and noticeably, with the road, hillside, and distant scenery evolving as the video progresses.
```

### `car2.png`

**C2-A — Drive in Frame**

```text
The white SUV drives slowly toward the right along the road, maintaining its shape and orientation. The vehicle remains fully visible throughout the video. The camera is fixed, with no panning or zooming.
```

**C2-B — Drive Exit**

```text
The white SUV accelerates toward the right along the road and completely exits the frame within the first half of the video. The camera remains fixed on the empty road and forest background after the vehicle disappears.
```

**C2-C — Camera Motion (Optional)**


### `bird1.png`

**B1-A — Flapping Wings**

```text
The colorful green bird remains perched on the branch and repeatedly flaps its wings while keeping its feet on the branch. Its body stays near the same position, with visible wing deformation. The camera remains completely stationary.
```

**B1-B — Fly Exit**

```text
The colorful green bird spreads its wings, takes off from the branch, and flies rapidly upward and to the right. The bird completely exits the frame within the first half of the video. The camera remains fixed on the empty branch afterward.
```

### `bird2.png`

**B2-A — Flapping Wings**

```text
The small brown bird stays perched on the broken branch while opening and flapping its wings several times. Its feet remain on the branch and its body stays near the original position. The camera remains stationary.
```

**B2-B — Fly Exit**

```text
The small brown bird takes off from the broken branch, flapping its wings rapidly as it flies upward and to the right. The bird completely disappears beyond the frame within the first half of the video. The camera stays fixed on the empty branch afterward.
```

## Single-Person Non-Rigid Motion

### `dance_person1.png`

**DP1-A — Simple Motion**

```text
The dancer in yellow continues a simple rhythmic movement, making small repeated steps and gentle arm movements near the original position. The dancer remains visible, without large rotations or complex pose changes. The camera stays fixed.
```

**DP1-B — Complex Dance**

```text
The dancer in yellow performs an energetic freestyle dance with rapid arm swings, deep knee bends, twisting torso movements, and changing leg positions. The dancer stays near the center of the frame. The camera remains completely stationary.
```

### `dance_person2.png`

**DP2-A — Simple Motion**

```text
The ballet dancer performs a slow, controlled movement, gently lowering her raised leg and extending her arms while maintaining balance near the original position. No large body rotation or displacement. The camera remains fixed.
```

**DP2-B — Complex Dance**

```text
The ballet dancer performs a dynamic sequence of pirouettes, high leg extensions, rapid turns, and deep body bends. Her arms and legs continuously change position while she remains within the stage area. The camera stays stationary.
```

## Multi-Person Motion and Occlusion

### `people.png`

**P-A — Multi-Person Motion**

```text
Everyone in the group smiles and waves their hands independently, with small and slightly different arm movements. All people remain standing in their original positions, without walking or crossing paths. The camera stays fixed.
```

### `soccer1.png`

**S1-A — Complex Multi-Player Motion**

```text
Several soccer players run rapidly in different directions while competing for the ball. Some players cross paths, briefly blocking one another from view as they turn and change direction. The camera remains fixed on the field.
```

### `soccer2.png`

**S2-A — Complex Multi-Player Motion**

```text
The player in red rapidly dribbles the ball while several players in blue rush toward him from different directions. Players cross paths, overlap, and briefly occlude one another during the challenge. The camera stays completely stationary.
```

### `dance_group1.png`

**DG1-A — Asynchronous Complex Motion**

```text
The dancers perform different energetic dance moves simultaneously. Some turn left while others turn right, with rapid arm swings, body twists, and leg movements. Several dancers cross paths and briefly occlude each other. The camera stays fixed.
```

### `dance_group2.png`

**DG2-A — Asynchronous Complex Motion**

```text
The ballet dancers perform complex movements independently, with some spinning clockwise while others turn counterclockwise. They execute high leg extensions, move across the stage, and briefly overlap one another. The camera stays fixed.
```

## Object Translation, Scaling, and Rotation

### `thing1.png`

**T1-A — Translation**

```text
The red apple smoothly slides horizontally toward the right across the dark surface without rotating or changing its apparent size. Its shape and appearance remain unchanged. The camera stays completely stationary.
```

**T1-B — Scale Up**

```text
The red apple moves smoothly toward the camera, becoming progressively larger in the frame while maintaining its shape and orientation. The camera remains completely fixed, with no zooming or panning.
```

**T1-C — Scale Down**

```text
The red apple moves smoothly away from the camera, becoming progressively smaller in the frame while maintaining its shape and orientation. The camera remains completely fixed, with no zooming or panning.
```

### `thing2.png`

**T2-A — Translation**

```text
The hand holding the tennis ball moves smoothly toward the right, keeping the ball firmly gripped. The ball maintains its original size, shape, and orientation, with minimal finger movement. The camera remains stationary.
```

**T2-B — Scale Up**

```text
The tennis ball moves smoothly toward the camera, becoming progressively larger in the frame while maintaining its shape and orientation. The camera remains completely fixed, with no zooming or panning.
``

**T1-C — Scale Down**

```text
The tennis ball moves smoothly away from the camera, becoming progressively smaller in the frame while maintaining its shape and orientation. The camera remains completely fixed, with no zooming or panning.
```

