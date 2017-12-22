@ echo security center
sc config wscsvc start= auto
net start wscsvc

@ echo volumn shadow copy
sc config vss start= auto
net start vss

@ echo micorsoft shadow copy provider
sc config swprv start= auto
net start swprv

@ echo micorsoft remote desktop 
sc config TermService start= auto
net start TermService


pause
