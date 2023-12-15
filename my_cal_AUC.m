function [pdpf,pftau,pdtau]=my_cal_AUC(r,normal_map,anomaly_map)
r=hyperNormalize(r);
r_max = max(r(:));
taus = linspace(0, r_max, 5000);
PF=zeros(1,5000); 
PD=zeros(1,5000);
for index1 = 1:length(taus)
  tau = taus(index1);
  anomaly_map_rx = (r(1,:)> tau);
  PF(index1) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD(index1) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
pdpf = sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
pftau=sum((PF(1:end-1)-PF(2:end)).*(taus(2:end)+taus(1:end-1))/2);
pdtau=sum((PD(1:end-1)-PD(2:end)).*(taus(2:end)+taus(1:end-1))/2);