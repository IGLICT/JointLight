import torch, torchvision, imageio, os, json, pandas, orjson
import imageio.v3 as iio
from PIL import Image
import pandas as pd
import random
import cv2
from pathlib import Path


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data, **kwargs):
        for operator in self.operators:
            data = operator(data, **kwargs)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data, **kwargs):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data, **kwargs):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str, **kwargs):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width


    def __call__(self, data: Image.Image, **kwargs):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data, **kwargs):
        return [data]


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, **kwargs):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames

class LoadVideoStartEnd(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str, **kwargs):
        reader = imageio.get_reader(data)
        max_num_frames = self.get_num_frames(reader)
        start_frame = kwargs.get('start_frame', 0)
        end_frame = kwargs.get('end_frame', max_num_frames)
        if max_num_frames < end_frame:
            end_frame -= ((end_frame - max_num_frames) // self.time_division_factor+1) * self.time_division_factor
        frames = []
        for frame_id in range(start_frame, end_frame):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data, **kwargs):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path, **kwargs):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str, **kwargs):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data, **kwargs)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data, **kwargs):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data, **kwargs)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data, **kwargs):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data, **kwargs):
        return os.path.join(self.base_path, data)


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        max_num_frames,
        time_division_factor=4,
        time_division_remainder=1,
        base_path=None, metadata_path=None, metadata_match_path=None,
        hdri_base_path=None, shading_cue_base_path=None, shading_gt_base_path=None,
        dataset_type=None,
        shading_cue_as_input=False,
        start_match_shading=0,
        end_match_shading=3000,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        dynamic_length_data_operator=lambda x: x,
        shading_cue_data_operator=lambda x: x,
        shading_gt_data_operator=lambda x: x,
        special_operator_map=None,
        load_from_cache=False,
        dynamic_length=False,
        uniform_input=False,
    ):
        self.max_num_frames = max_num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.metadata_match_path = metadata_match_path
        self.hdri_base_path = hdri_base_path
        self.shading_cue_base_path = shading_cue_base_path
        self.shading_gt_base_path = shading_gt_base_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.dynamic_length_data_operator = dynamic_length_data_operator
        self.shading_cue_data_operator = shading_cue_data_operator
        self.shading_gt_data_operator = shading_gt_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.data_match = []
        self.cached_data = []
        self.load_from_cache = load_from_cache
        self.dynamic_length = dynamic_length
        self.uniform_input = uniform_input
        self.load_metadata(metadata_path)
        if shading_cue_as_input:
            self.load_metadata_match(metadata_match_path)
        self.dataset_type = dataset_type
        self.shading_cue_as_input = shading_cue_as_input
        self.start_match_shading = start_match_shading
        self.end_match_shading = end_match_shading
        self.current_step = 0
    
    def set_step(self, step):
        self.current_step = step

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])


    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])

    @staticmethod
    def dynamic_length_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideoStartEnd(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])


    @staticmethod
    def cached_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
    
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".wan2.2tensors.pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        self.search_for_cached_data_files(self.base_path)
        print(f"{len(self.cached_data)} cached data files found.")
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "rb") as f:  # 注意：orjson 需要 bytes
                metadata = orjson.loads(f.read())
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            self.metadata = metadata

    def load_metadata_match(self, metadata_match_path):
        self.search_for_cached_data_files(self.base_path)
        print(f"{len(self.cached_data)} cached data files found.")
        if metadata_match_path is None:
            print("No metadata_match_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_match_path.endswith(".json"):
            with open(metadata_match_path, "rb") as f:  # 注意：orjson 需要 bytes
                metadata_match = orjson.loads(f.read())
            self.data_match = metadata_match
        elif metadata_match_path.endswith(".jsonl"):
            metadata_match = []
            with open(metadata_match_path, 'r') as f:
                for line in f:
                    metadata_match.append(json.loads(line.strip()))
            self.data_match = metadata_match
        else:
            metadata_match = pandas.read_csv(metadata_match_path)
            self.data_match = [metadata_match.iloc[i].to_dict() for i in range(len(metadata_match))]
            self.metadata_match = metadata_match

    def get_num_frames(self):
        num_frames = self.max_num_frames
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
        return num_frames

    def get_match_prob(self):
        return min(0.5, 0.5 * max(0, self.current_step - self.start_match_shading) / (self.end_match_shading - self.start_match_shading)) if self.shading_cue_as_input else 0

    def get_dynamic_lighting_prob(self):
        return 0.2 if self.dataset_type == 3 else 0

    def __getitem__(self, data_id):
        while True:
            try:
                match_shading = random.random() < self.get_match_prob()
                dynamic_lighting = random.random() < self.get_dynamic_lighting_prob()
                if match_shading:
                    data = self.data_match[data_id % len(self.data_match)].copy()
                    shading_cue_path = os.path.join(self.shading_cue_base_path, data['video'])
                    data['shading_cue_path'] = shading_cue_path
                    data['shading_cue'] = data['video']
                else:
                    data = self.data[data_id % len(self.data)].copy()
                    data['shading_cue_path'] = None
                    data['shading_cue'] = None
                data['match_shading'] = match_shading
                input_path = os.path.join(self.base_path, data['video'])
                shading_gt_path = os.path.join(self.shading_gt_base_path, data['video'])
                data['input_video_path'] = input_path
                data['shading_gt_path'] = shading_gt_path
                data['shading_gt'] = data['video']
                data["load_from_cache"] = self.load_from_cache
                data["data_type"] = self.dataset_type

                input_name = os.path.basename(input_path)
                if 'Unlit' in input_name:
                    continue

                if self.dataset_type in [0, 1, 3, 4]: # paired ue image/video datset
                    all_names = [name for name in os.listdir(os.path.dirname(input_path)) if name != 'Unlit.mp4' and (name.endswith('.mp4') or name.endswith('.png'))]
                    all_names.remove(input_name)
                    cond_name = 'all_white.mp4' if self.uniform_input else random.choice(all_names)
                    all_names.remove(cond_name)
                    cond_path = os.path.join(os.path.dirname(input_path), cond_name)
                    cond_video = cond_path.split(str(Path(self.base_path))+'/')[1]
                    data['cond_video'] = cond_video
                    data['cond_video_path'] = cond_path
                    
                elif self.dataset_type == 6:
                    cond_path = os.path.join(os.path.dirname(input_path), 'input', 'deg.mp4')
                    cond_video = cond_path.split(str(Path(self.base_path))+'/')[1]
                    data['cond_video_path'] = cond_path
                    data['cond_video'] = cond_video
                    if not os.path.exists(cond_path):
                        data['cond_video'] = None

                    data['shading_gt_path'] = os.path.join(os.path.dirname(input_path), 'shading.mp4')
                    data['shading_gt'] = data['shading_gt_path'].split(str(Path(self.base_path))+'/')[1]
                    data['shading_cue_path'] = None
                    data['shading_cue'] = None
                    data['match_shading'] = False
                else:
                    data['cond_video'] = None
                    data['cond_video_path'] = None

                if self.load_from_cache:
                    input_cached_latents_path = input_path + ".wan2.2tensors.pth"
                    cond_cached_latents_path = cond_path + ".wan2.2tensors.pth"
                    input_cached_latents = self.cached_data_operator(input_cached_latents_path)
                    cond_cached_latents = self.cached_data_operator(cond_cached_latents_path)
                    data.update({'context': input_cached_latents['prompt_emb'], 'input_latents': input_cached_latents['latents'], 'cond_latents': cond_cached_latents['latents']})
                if self.dynamic_length:
                    start_frame = random.choice(range(0, self.max_num_frames-self.time_division_remainder, self.time_division_factor))
                    end_frame = random.choice(range(start_frame+self.time_division_remainder, self.max_num_frames+1, self.time_division_factor))
                else:
                    start_frame = 0
                    end_frame = self.get_num_frames()
                if random.random() < 0.2: # image
                    start_frame = 0
                    end_frame = 1
                        
                if random.random() < 0.12:
                    data['cond_video'] = None
                    data['cond_video_path'] = None

                for key in self.data_file_keys:
                    if key in data:
                        if key in self.special_operator_map:
                            data[key] = self.special_operator_map[key]
                        elif key in self.data_file_keys:
                            if key in ['shading_cue']:
                                if data[key] is not None:
                                    data[key] = self.shading_cue_data_operator(data[key], start_frame=start_frame, end_frame=end_frame)
                            elif key in ['shading_gt']:
                                data[key] = self.shading_gt_data_operator(data[key], start_frame=start_frame, end_frame=end_frame)
                            elif key in ['video', 'cond_video']:
                                if data[key] is not None:
                                    data[key] = self.dynamic_length_data_operator(data[key], start_frame=start_frame, end_frame=end_frame)
                            else:
                                data[key] = self.main_data_operator(data[key])


                assert len(data['video']) == len(data['shading_gt'])
                if data['cond_video'] is not None: assert len(data['video']) == len(data['cond_video'])
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                data_id += 1
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
