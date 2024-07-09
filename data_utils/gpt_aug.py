# USC_GPT_AUG = {
#     "Walking Forward": "Advancing at a consistent pace on foot. This activity involves moving forward with one foot in front of the other, maintaining balance and coordination.",
#     "Walking Left": "Shifting to the left at a steady pace on foot. This activity requires lateral movement while maintaining stability and direction.",
#     "Walking Right": "Moving to the right at an even pace on foot. This involves lateral movement with a focus on maintaining a straight path and balance.",
#     "Walking Upstairs": "Ascending stairs steadily and consistently. This activity involves lifting each foot to climb steps while maintaining balance and coordination.",
#     "Walking Downstairs": "Descending stairs at a constant pace. This activity involves stepping down each stair while focusing on balance and control.",
#     "Running Forward": "Sprinting rapidly ahead on foot. This activity involves moving at a faster pace than walking, with both feet leaving the ground during each stride.",
#     "Jumping Up": "Launching oneself upwards from the ground. This activity involves propelling the body off the ground using leg strength and coordination.",
#     "Sitting": "Seated with weight supported on buttocks and thighs. This activity involves resting in a chair or on a surface with a relaxed posture.",
#     "Standing": "Maintaining an erect posture on feet without aid. This activity involves standing upright, supporting body weight on both feet without leaning or sitting.",
#     "Sleeping": "Reclining in a flat position for repose. This activity involves resting in a horizontal position on a bed or similar surface for relaxation and sleep.",
#     "Elevator Up": "Rising vertically in an elevator. This activity involves moving upwards in a confined space, typically using an elevator.",
#     "Elevator Down": "Sinking vertically in an elevator. This activity involves moving downwards in a confined space, typically using an elevator."
# }

USC_GPT_AUG = {
    "Walking Forward": "Linear movement in the forward direction. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs.",

    "Walking Left": "Linear movement to the left. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs.",

    "Walking Right": "Linear movement to the right. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs.",

    "Walking Upstairs": "Vertical movement upwards using stairs. Speed is slower than walking on a flat surface. Characterized by lifting legs higher and consistent arm movement for balance.",

    "Walking Downstairs": "Vertical movement downwards using stairs. Speed is slower than walking on a flat surface. Characterized by placing the foot on each step cautiously and using railing for support.",

    "Running Forward": "Linear movement in the forward direction. Speed is faster than walking. Characterized by a faster and more forceful stride, with both feet leaving the ground.",

    "Jumping Up": "Vertical movement upwards without forward motion. Characterized by both feet leaving the ground simultaneously and arms usually swinging upwards.",

    "Sitting": "Static posture with no movement. Characterized by a seated position where the weight is supported by a surface.",

    "Standing": "Static posture with no movement. Characterized by an upright position where the weight is supported by the feet.",

    "Sleeping": "Static posture with no movement. Characterized by lying down, eyes closed, and little to no movement.",

    "Elevator Up": "Vertical movement upwards in an elevator. Characterized by a smooth and consistent upward motion without the need for physical exertion.",

    "Elevator Down": "Vertical movement downwards in an elevator. Characterized by a smooth and consistent downward motion without the need for physical exertion."
}


mmwave_GPT_AUG = {
    "Stretching and relaxing": "Dynamic movement involving stretching and then relaxing muscles. Characterized by elongation of the body and subsequent release of tension.",
    
    "Chest expansion(horizontal)": "Dynamic movement expanding the chest horizontally. Characterized by opening the chest and extending the arms sideways.",
    
    "Chest expansion (vertical)": "Dynamic movement expanding the chest vertically. Characterized by lifting the arms upwards and expanding the chest.",
    
    "Twist (left)": "Rotational movement to the left. Characterized by twisting the torso and possibly the hips to the left.",
    
    "Twist (right)": "Rotational movement to the right. Characterized by twisting the torso and possibly the hips to the right.",
    
    "Mark time": "Dynamic movement involving marching in place. Characterized by lifting and lowering the feet alternately without forward motion.",
    
    "Limb extension (left)": "Dynamic movement extending the left limb. Characterized by stretching the left arm or leg outward.",
    
    "Limb extension (right)": "Dynamic movement extending the right limb. Characterized by stretching the right arm or leg outward.",
    
    "Lunge (toward left-front)": "Dynamic movement lunging forward and to the left. Characterized by stepping forward with the left leg and bending both knees.",
    
    "Lunge (toward right-front)": "Dynamic movement lunging forward and to the right. Characterized by stepping forward with the right leg and bending both knees.",
    
    "Limb extension (both)": "Dynamic movement extending both limbs simultaneously. Characterized by stretching both arms or legs outward.",
    
    "Squat": "Dynamic movement lowering the body by bending knees and hips. Characterized by lowering the body towards the ground while keeping the back straight.",
    
    "Raising hand (left)": "Dynamic movement raising the left hand upwards. Characterized by lifting the left arm towards the shoulder or above the head.",
    
    "Raising hand (right)": "Dynamic movement raising the right hand upwards. Characterized by lifting the right arm towards the shoulder or above the head.",
    
    "Lunge (toward left side)": "Dynamic movement lunging sideways to the left. Characterized by stepping sideways with the left leg and bending the knee.",
    
    "Lunge (toward right side)": "Dynamic movement lunging sideways to the right. Characterized by stepping sideways with the right leg and bending the knee.",
    
    "Waving hand (left)": "Dynamic movement waving the left hand. Characterized by moving the left hand back and forth in a waving motion.",
    
    "Waving hand (right)": "Dynamic movement waving the right hand. Characterized by moving the right hand back and forth in a waving motion.",
    
    "Picking up things": "Dynamic movement bending down to pick up objects. Characterized by lowering the body and reaching towards the ground.",
    
    "Throwing (toward left side)": "Dynamic movement throwing an object to the left side. Characterized by extending the arm and releasing the object.",
    
    "Throwing (toward right side)": "Dynamic movement throwing an object to the right side. Characterized by extending the arm and releasing the object.",
    
    "Kicking (toward left side)": "Dynamic movement kicking towards the left side. Characterized by extending the left leg outward.",
    
    "Kicking (toward right side)": "Dynamic movement kicking towards the right side. Characterized by extending the right leg outward.",
    
    "Body extension (left)": "Dynamic movement extending the left side of the body. Characterized by stretching and lengthening the left side.",
    
    "Body extension (right)": "Dynamic movement extending the right side of the body. Characterized by stretching and lengthening the right side.",
    
    "Jumping up": "Vertical movement upwards. Characterized by both feet leaving the ground simultaneously.",
    
    "Bowing": "Dynamic movement bending forward from the waist as a sign of respect or acknowledgment. Characterized by lowering the head and upper body towards the ground."
}

wifi_GPT_AUG = mmwave_GPT_AUG
# mmwave_GPT_AUG = {
#     "Stretching and relaxing": "The individual performs elongation and subsequent relaxation of their muscles, alternating between tension and release.",
#     "Chest expansion(horizontal)": "The person expands their chest in a horizontal plane, emphasizing a broadening of the chest area.",
#     "Chest expansion (vertical)": "The individual expands their chest in a vertical direction, lifting the chest upwards.",
#     "Twist (left)": "The person rotates their upper body to the left, creating a torsional movement on the horizontal plane.",
#     "Twist (right)": "The individual rotates their upper body to the right, inducing a torsional movement on the horizontal plane.",
#     "Mark time": "The individual marches in place, lifting and lowering their feet alternately without moving forward.",
#     "Limb extension (left)": "The person stretches out their left limb, reaching and extending it outward.",
#     "Limb extension (right)": "The individual stretches out their right limb, reaching and extending it outward.",
#     "Lunge (toward left-front)": "The person takes a step forward and to the left, bending the front knee while keeping the back leg straight.",
#     "Lunge (toward right-front)": "The individual takes a step forward and to the right, bending the front knee while keeping the back leg straight.",
#     "Limb extension (both)": "The person simultaneously stretches out both limbs, reaching and extending them outward.",
#     "Squat": "The individual lowers their body by bending their knees and hips, keeping their feet flat on the ground.",
#     "Raising hand (left)": "The person lifts their left hand upwards, extending it above the shoulder level.",
#     "Raising hand (right)": "The individual lifts their right hand upwards, extending it above the shoulder level.",
#     "Lunge (toward left side)": "The person steps to the side and to the left, bending the knee and maintaining balance.",
#     "Lunge (toward right side)": "The individual steps to the side and to the right, bending the knee and maintaining balance.",
#     "Waving hand (left)": "The person moves their left hand in a waving motion, typically in a horizontal plane.",
#     "Waving hand (right)": "The individual moves their right hand in a waving motion, typically in a horizontal plane.",
#     "Picking up things": "The individual bends down or reaches out to grab objects from the ground or a lower level.",
#     "Throwing (toward left side)": "The person throws an object towards the left side, extending the arm in a forward motion.",
#     "Throwing (toward right side)": "The individual throws an object towards the right side, extending the arm in a forward motion.",
#     "Kicking (toward left side)": "The person kicks towards the left side, extending the leg in a forward motion.",
#     "Kicking (toward right side)": "The individual kicks towards the right side, extending the leg in a forward motion.",
#     "Body extension (left)": "The person extends their left side of the body, stretching and lengthening it.",
#     "Body extension (right)": "The individual extends their right side of the body, stretching and lengthening it.",
#     "Jumping up": "The individual propels themselves vertically off the ground, executing a brief aerial action.",
#     "Bowing": "The person bends forward from the waist as a sign of respect or acknowledgment."
# }


# mmwave_GPT_AUG = {
#     "Stretching and relaxing": "Engaging in stretching movements followed by relaxation. This activity involves extending muscles to increase flexibility and then releasing tension to relax.",
#     "Chest expansion(horizontal)": "Expanding the chest horizontally. This activity involves stretching the chest muscles outward while maintaining a horizontal posture.",
#     "Chest expansion (vertical)": "Expanding the chest vertically. This activity involves lifting the chest upwards to stretch the chest muscles vertically.",
#     "Twist (left)": "Twisting the body to the left. This activity involves rotating the torso to the left while maintaining balance and stability.",
#     "Twist (right)": "Twisting the body to the right. This activity involves rotating the torso to the right while maintaining balance and stability.",
#     "Mark time": "Marching in place without moving forward. This activity involves lifting and lowering the legs alternately while staying in one position.",
#     "Limb extension (left)": "Extending the left limb outward. This activity involves stretching the left arm or leg to its full length.",
#     "Limb extension (right)": "Extending the right limb outward. This activity involves stretching the right arm or leg to its full length.",
#     "Lunge (toward left-front)": "Performing a lunge toward the left-front direction. This activity involves stepping forward and bending the front knee while keeping the back leg straight.",
#     "Lunge (toward right-front)": "Performing a lunge toward the right-front direction. This activity involves stepping forward and bending the front knee while keeping the back leg straight.",
#     "Limb extension (both)": "Simultaneously extending both limbs outward. This activity involves stretching both arms or legs to their full lengths at the same time.",
#     "Squat": "Bending the knees and lowering the body. This activity involves squatting down by bending the knees while keeping the back straight.",
#     "Raising hand (left)": "Raising the left hand upwards. This activity involves lifting the left arm vertically to raise the hand.",
#     "Raising hand (right)": "Raising the right hand upwards. This activity involves lifting the right arm vertically to raise the hand.",
#     "Lunge (toward left side)": "Performing a lunge toward the left side. This activity involves stepping to the left and bending the knee while keeping the other leg straight.",
#     "Lunge (toward right side)": "Performing a lunge toward the right side. This activity involves stepping to the right and bending the knee while keeping the other leg straight.",
#     "Waving hand (left)": "Waving the left hand in the air. This activity involves moving the left hand back and forth in a waving motion.",
#     "Waving hand (right)": "Waving the right hand in the air. This activity involves moving the right hand back and forth in a waving motion.",
#     "Picking up things": "Bending down to pick up objects from the ground. This activity involves lowering the body to reach and lift items.",
#     "Throwing (toward left side)": "Throwing an object toward the left side. This activity involves propelling an object with the left hand or arm in a throwing motion.",
#     "Throwing (toward right side)": "Throwing an object toward the right side. This activity involves propelling an object with the right hand or arm in a throwing motion.",
#     "Kicking (toward left side)": "Kicking an object or target toward the left side. This activity involves extending the left leg to make contact with an object.",
#     "Kicking (toward right side)": "Kicking an object or target toward the right side. This activity involves extending the right leg to make contact with an object.",
#     "Body extension (left)": "Extending the left side of the body outward. This activity involves stretching the left arm or leg to its full length.",
#     "Body extension (right)": "Extending the right side of the body outward. This activity involves stretching the right arm or leg to its full length.",
#     "Jumping up": "Propelling oneself off the ground vertically. This activity involves using leg strength to jump upwards.",
#     "Bowing": "Bending forward at the waist as a gesture of respect or acknowledgment. This activity involves lowering the upper body while keeping the back straight."
# }


pamap_GPT_AUG = action_features = {
    "Lying": "Complete rest position with no movement. Characterized by lying flat on the back or side, eyes closed, relaxed muscles, and a state of relaxation.",

    "Sitting": "Static posture with minimal movement. Characterized by a seated position where the weight is supported by a surface, legs are bent at the knees, and feet are on the ground or elevated.",

    "Standing": "Static posture with no movement. Characterized by an upright position where the weight is supported by the feet, legs are straight, and arms are relaxed by the sides.",

    "Walking": "Linear movement in a forward direction. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs, swinging arms, and continuous steps.",

    "Running": "Linear movement in a forward direction. Speed is faster than walking. Characterized by a rapid and forceful stride, both feet leaving the ground simultaneously, and intense arm movement.",

    "Cycling": "Linear movement using a bicycle. Speed can vary. Characterized by circular leg movements, continuous pedaling, hand grip on handlebars, and leaning forward or upright posture.",

    "Nordic Walking": "Linear movement using poles for support. Speed is consistent but slower than running. Characterized by the use of poles to aid in walking, rhythmic arm swinging, and upright posture.",

    "Ascending Stairs": "Vertical movement upwards using stairs. Speed is slower than walking on a flat surface. Characterized by lifting legs higher, gripping the railing for support, and continuous step-by-step movement.",

    "Descending Stairs": "Vertical movement downwards using stairs. Speed is slower than walking on a flat surface. Characterized by placing the foot on each step cautiously, leaning slightly backward, and using railing for support.",

    "Vacuum cleaning": "Linear movement while cleaning using a vacuum cleaner. Speed is consistent and can vary based on the cleaning area. Characterized by pushing and pulling motions, back and forth movements, and hand grip on the vacuum handle.",

    "Ironing": "Static posture with repetitive movement. Characterized by standing in front of an ironing board, moving an iron back and forth over clothes, rotating the wrist, and maintaining a focused posture.",

    "Rope Jumping": "Vertical movement upwards and downwards using a jump rope. Speed can vary. Characterized by jumping over a rope that passes under the feet, swinging the rope with wrists, and maintaining a rhythmic jumping pattern."
}


widar_GPT_AUG  = {
    "Push and Pull": "Dynamic movement involving exerting force to move an object or surface either towards or away from oneself. Characterized by alternating actions of pushing and pulling.",

    "Sweep": "Linear movement used for cleaning using a broom or brush. Characterized by sweeping motions back and forth to gather and collect debris or dirt.",

    "Clap": "Dynamic movement of bringing hands together with force to create a sound. Characterized by the palms of the hands meeting quickly and producing a clapping noise.",

    "Slide": "Horizontal movement while maintaining contact with a surface, without lifting the feet. Characterized by smooth and gliding motions across a surface.",

    "Draw Circle": "Dynamic movement tracing a circle shape. Characterized by a continuous hand movement creating a circular shape on a surface.",

    "Draw Zigzag": "Dynamic movement tracing a zigzag pattern. Characterized by a continuous hand movement creating a zigzag shape on a surface."
}



# pamap_GPT_AUG = {
#     "lying": "Resting in a horizontal position on a flat surface. This activity involves reclining the body to rest and relax.",
#     "sitting": "Being in a seated position with the weight supported by the buttocks and thighs. This activity involves resting in a chair or on a surface.",
#     "standing": "Maintaining an upright position on the feet without any external support. This activity involves supporting the body weight on both feet.",
#     "walking": "Moving at a regular pace by taking steps with alternate feet touching the ground. This activity involves propelling oneself forward on foot.",
#     "running": "Moving rapidly on foot by taking quick strides and often at a pace faster than walking. This activity involves sprinting with both feet leaving the ground during each stride.",
#     "cycling": "Riding a bicycle, typically using pedals and a chain to propel oneself forward. This activity involves pedaling to move forward on a bicycle.",
#     "Nordic walking": "A fitness activity that involves walking with the help of specially designed poles, engaging both the upper and lower body. This activity involves using poles to enhance the walking motion.",
#     "ascending stairs": "Moving upward by stepping on a series of elevated platforms or steps. This activity involves lifting each foot to climb steps.",
#     "descending stairs": "Moving downward by stepping down from a series of elevated platforms or steps. This activity involves stepping down each stair while focusing on balance and control.",
#     "vacuum cleaning": "Using a vacuum cleaner to remove dirt, dust, and debris from floors and other surfaces. This activity involves pushing and maneuvering a vacuum cleaner.",
#     "ironing": "Using a heated iron to remove wrinkles from clothing or fabric by pressing. This activity involves sliding an iron over fabric to smooth out wrinkles.",
#     "rope jumping": "Exercising by jumping over a rope that is swung over the head and under the feet continuously. This activity involves jumping over a rope with coordinated timing."
# }


uthar_GPT_AUG = {
    "Lie down": "Transitioning from a standing or sitting position to a horizontal posture on a flat surface. Characterized by lowering the body gradually, usually supported by hands or elbows.",

    "Fall": "Involuntary movement resulting in a sudden drop to the ground or a lower surface. Characterized by an unexpected loss of balance or support, leading to a rapid descent.",

    "Walk": "Linear movement in a forward direction. Speed is consistent but slower than running. Characterized by alternating movement of left and right legs, swinging arms, and continuous steps.",

    "Pick up": "Dynamic movement involving bending down to lift an object from the ground or a lower surface. Characterized by lowering the body, reaching for the object, and then lifting it upwards.",

    "Run": "Linear movement in a forward direction. Speed is faster than walking. Characterized by a rapid and forceful stride, both feet leaving the ground simultaneously, and intense arm movement.",

    "Sit down": "Transitioning from a standing or walking position to a seated posture on a chair, bench, or the ground. Characterized by bending the knees, lowering the body, and positioning oneself on a surface.",

    "Stand up": "Transitioning from a seated or lying position to an upright standing posture. Characterized by pushing off the ground or surface with hands or arms, straightening the legs, and rising to a standing position."
}



GPT_AUG_DICT = {
    "USC": USC_GPT_AUG,
    "mmwave": mmwave_GPT_AUG,
    "pamap": pamap_GPT_AUG,
    "wifi": mmwave_GPT_AUG,
    "lidar": mmwave_GPT_AUG,
    "widar": widar_GPT_AUG,
    "uthar": uthar_GPT_AUG

}

