Command to run with VIDEO

python scripts/detect_crop_upright_read_video.py --model "C:/Users/Gnanasekar/Downloads/processtest1/models/detect/best.pt" --image "C:/Users/Gnanasekar/Downloads/processtest1/inputs/videos/MicrosoftTeams-video.mp4" --upright_model "C:/Users/Gnanasekar/Downloads/processtest1/models/upright/best.pt" --digit_model "C:/Users/Gnanasekar/Downloads/processtest1/models/read/best3.pt" --out "C:/Users/Gnanasekar/Downloads/processtest1/outputs/crops" --upright_out "C:/Users/Gnanasekar/Downloads/processtest1/outputs/upright" --digits_out "C:/Users/Gnanasekar/Downloads/processtest1/outputs/digits" --conf 0.1 --frame_step 5


Command to run with IMAGE

python scripts/detect_crop_upright_read_image.py --model "C:/Users/Gnanasekar/Downloads/processtest1/models/detect/best.pt" --image "C:/Users/Gnanasekar/Downloads/processtest1/inputs/videos/MicrosoftTeams-video.mp4" --upright_model "C:/Users/Gnanasekar/Downloads/processtest1/models/upright/best.pt" --digit_model "C:/Users/Gnanasekar/Downloads/processtest1/models/read/best3.pt" --out "C:/Users/Gnanasekar/Downloads/processtest1/outputs/crops" --upright_out "C:/Users/Gnanasekar/Downloads/processtest1/outputs/upright" --digits_out "C:/Users/Gnanasekar/Downloads/processtest1/outputs/digits" --conf 0.1                

Command to run with IMAGES

python scripts/detect_crop_upright_read_images.py --images_folder "C:/Users/Gnanasekar/Downloads/processtest1/inputs/images" --out "C:/Users/Gnanasekar/Downloads/processtest1/outputs" --model "C:/Users/Gnanasekar/Downloads/processtest1/models/detect/best.pt" --upright_model "C:/Users/Gnanasekar/Downloads/processtest1/models/upright/best.pt" --digit_model "C:/Users/Gnanasekar/Downloads/processtest1/models/read/best.pt" --conf 0.1
