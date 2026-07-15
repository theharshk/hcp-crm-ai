import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { api } from "../api/api";

export const fetchHcps = createAsyncThunk("interactions/fetchHcps", async () => {
  return api.listHcps();
});

export const createHcp = createAsyncThunk("interactions/createHcp", async (payload) => {
  return api.createHcp(payload);
});

export const fetchHistory = createAsyncThunk("interactions/fetchHistory", async (hcpId) => {
  return api.listInteractions(hcpId);
});

export const submitStructuredInteraction = createAsyncThunk(
  "interactions/submitStructured",
  async (payload) => {
    const interaction = await api.createInteraction(payload);
    const compliance = await api.checkCompliance(interaction.id);
    return { interaction, compliance };
  }
);

export const editInteraction = createAsyncThunk(
  "interactions/edit",
  async ({ id, payload }) => {
    return api.updateInteraction(id, payload);
  }
);

const interactionsSlice = createSlice({
  name: "interactions",
  initialState: {
    hcps: [],
    selectedHcpId: null,
    history: [],
    lastSubmission: null,
    selectedInteraction: null,
    draftInteraction: null,
    draftHcp: null,
    complianceWarnings: [],
    status: "idle",
    error: null,
  },
  reducers: {
    selectHcp(state, action) {
      state.selectedHcpId = action.payload;
      state.selectedInteraction = null;
      state.lastSubmission = null;
      state.draftInteraction = null;
      state.draftHcp = null;
    },
    selectInteraction(state, action) {
      state.selectedInteraction = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchHcps.fulfilled, (state, action) => {
        state.hcps = action.payload;
      })
      .addCase(createHcp.fulfilled, (state, action) => {
        state.hcps.push(action.payload);
        state.selectedHcpId = action.payload.id;
        state.selectedInteraction = null;
        state.lastSubmission = null;
        state.draftInteraction = null;
        state.draftHcp = null;
        state.history = [];
      })
      .addCase(fetchHistory.fulfilled, (state, action) => {
        state.history = action.payload;
      })
      .addCase(submitStructuredInteraction.pending, (state) => {
        state.status = "loading";
      })
      .addCase(submitStructuredInteraction.fulfilled, (state, action) => {
        state.status = "idle";
        state.lastSubmission = action.payload.interaction;
        state.selectedInteraction = action.payload.interaction;
        const warnings = JSON.parse(action.payload.compliance.result || "{}").warnings || [];
        state.complianceWarnings = warnings;
        state.history = [action.payload.interaction, ...state.history];
      })
      .addCase(submitStructuredInteraction.rejected, (state, action) => {
        state.status = "idle";
        state.error = action.error.message;
      })
      .addCase(editInteraction.fulfilled, (state, action) => {
        state.history = state.history.map((h) => (h.id === action.payload.id ? action.payload : h));
        if (state.selectedInteraction && state.selectedInteraction.id === action.payload.id) {
          state.selectedInteraction = action.payload;
        }
      })
      .addCase("chat/sendMessage/fulfilled", (state, action) => {
        const payloadState = action.payload.state;
        if (payloadState) {
          const isNewDoctorDraft = !!payloadState.draft_hcp;
          const isDoctorSwitched = payloadState.hcp && payloadState.hcp.id !== state.selectedHcpId;
          
          if (isNewDoctorDraft || isDoctorSwitched) {
            state.selectedInteraction = null;
            state.draftInteraction = null;
            state.complianceWarnings = [];
            // If a new doctor is being drafted (not yet in DB), clear the panel
            // so the old doctor's profile card doesn't confusingly remain on screen
            if (isNewDoctorDraft && !isDoctorSwitched) {
              state.selectedHcpId = null;
              state.history = [];
            }
          }
          if (payloadState.interaction) {
            state.lastSubmission = payloadState.interaction;
            state.selectedInteraction = payloadState.interaction;
            state.draftInteraction = null;
            state.draftHcp = null;
            // Also prepend to history if it's not already in there to keep lists synced
            if (!state.history.some((h) => h.id === payloadState.interaction.id)) {
              state.history = [payloadState.interaction, ...state.history];
            }
          } else if (payloadState.draft_interaction) {
            state.draftInteraction = payloadState.draft_interaction;
            state.selectedInteraction = null;
          } else {
            state.selectedInteraction = null;
            state.draftInteraction = null;
          }
          
          if (payloadState.draft_hcp) {
            state.draftHcp = payloadState.draft_hcp;
          } else if (payloadState.hcp) {
            state.draftHcp = null;
          }

          if (payloadState.hcp) {
            state.selectedHcpId = payloadState.hcp.id;
            state.draftHcp = null;
            const hcpIndex = state.hcps.findIndex((h) => h.id === payloadState.hcp.id);
            if (hcpIndex >= 0) {
              state.hcps[hcpIndex] = payloadState.hcp;
            } else {
              // New doctor created via chat — add to list
              state.hcps.push(payloadState.hcp);
            }
          }
          // NOTE: We intentionally do NOT clear selectedHcpId when payloadState.hcp is null
          // UNLESS a new draft_hcp is in progress (handled above), so that the panel goes blank
          // rather than keeping the old doctor visible.
          state.complianceWarnings = payloadState.compliance_warnings || [];
        }
      });
  },
});

export const { selectHcp, selectInteraction } = interactionsSlice.actions;
export default interactionsSlice.reducer;
