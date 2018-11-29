function [s] = getSpec(folder, filename)
%Get spectrogram of a single audio song
%   Refer to CNN for music emotion classification

nfft = 512;
window = 512;
M = nfft/2 + 1;

[y,Fs] = audioread(sprintf('%s/%s.mp3',folder, filename));
noverlap = ((M+1)*window - length(y))/M;
s = spectrogram(y, window, noverlap, nfft) ;
s = abs(s(:,1:257));
s = s.^2;
%dlmwrite(sprintf('spec_data/%s.csv',filename),s);


end

