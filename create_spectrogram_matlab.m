path = './CAL500_Song_32kps';
nfft = 512;
window = 512;
M = nfft/2 + 1;
listing = dir(path);

%%
for i=3:length(listing)
    [y,Fs] = audioread(sprintf('%s/%s',path,listing(i).name));
    noverlap = ((M+1)*window - length(y))/M;
    s = spectrogram(y, window, noverlap, nfft) ;
    s = abs(s(:,1:257)); % the last column of s is empty
    s = s.^2;
    dlmwrite(sprintf('spec_data/%s.csv',listing(i).name(1:end-4)),s);
end

