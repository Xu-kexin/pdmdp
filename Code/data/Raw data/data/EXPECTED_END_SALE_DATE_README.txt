SELECT SVVD, FCIL_CDE, BERTH_DEP_DT_LOC, SCH_STATE
FROM ITS_SCHEDULE
ORDER BY SVVD, FCIL_CDE, SCH_STATE

SVVD为航次
FCIL_CDE为出发码头的代号
BERTH_DEP_DT_LOC为发船时间
SCH_STATE中longterm表示该时间为预期发船时间，actual表示该时间为真实发船时间

由于时间中存在空缺，处理思路为：如果longterm时间不空缺，则用longterm时间作为发船时间
如果longterm时间空缺，则用actual时间作为发船时间