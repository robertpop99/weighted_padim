% folder = 'datasets/kilauea/';
folder = 'datasets/taal/';
subfolders = ["train"; "test"; "abnormal"];
% subfolders = ["test_abnormal"];
all_files_folder = [folder, 'all/'];

all_files = dir(all_files_folder);

sea = 0;
sea = imread([folder, 'taal_water_mask.tif'])==1;
% files_in_main_folder = dir(folder);
% for i = 1:length(files_in_main_folder)
%     file_name = files_in_main_folder(i).name;
%     if contains(file_name, 'mask_sea')
%         sea = imread([folder, file_name])==0;
%         break
%     end
% end

% water = imread('datasets/reykjanes/reykjanes_016A_02504_162118_mask_water.tif')==1;
% sea = imread([folder, 'lamongan_003D_09757_111111_mask_sea.tif'])==0;

file_type = 'unw.';
old_folders_name = '/unw/';
new_folders_name = '/unw_remade_water/';
new_file_extension = '.unw.png';

%mean = 1.149059263159326;
%std = 16.078874948699298;
% mean = 0.9586881064788321
% std = 21.663192715558232

for f = 1:length(subfolders)
    write_destination = [folder, convertStringsToChars(subfolders(f)), new_folders_name ];
    if ~isfolder(write_destination)
        mkdir(write_destination)
    end
    files = dir([folder, convertStringsToChars(subfolders(f)), old_folders_name]);
    for z = 1:length(files)
        file = files(z);
        if file.isdir
            continue
        end
        name = split(file.name,'.');
        name = name(1);
        name = split(name, 'geo');
        name = name(1);
        name = char(name);
        for i = 1:length(all_files)
            interferogram_name = all_files(i).name;
            if contains(interferogram_name, name) && contains(interferogram_name, file_type)
                img = imread([all_files_folder, interferogram_name]);
                
%                 img(isnan(img)) = 0;
                MASK = (img==0) | isnan(img) | sea;

                img = img - mean(img(~MASK));
                
                % MASK = MASK & ~sea;

                RES_fill = regionfill(img, imclose(imdilate(MASK,strel('disk',2)),strel('disk',5)));
                img(imdilate(MASK,strel('disk',2))>0) = RES_fill(imdilate(MASK,strel('disk',2))>0);

                % img(sea) = 0;
                img = max(0, min(1,(img + 30)/60));
                img(sea) = 0;

                % img(sea) = 0;
%                 img = (img + pi) / (2 * pi);
                % [write_destination, name, new_file_extension]
                imwrite(img, [write_destination, name, new_file_extension])
            end
        end

    end
end
