/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "meshing/processing_modifiers/export_visual_mesh.h"

#include "app/app.h" // AssetType, SuperDexStudio::GetFileDialogPath
#include "meshing/processing_modifiers/processing_export_path.h" // DefaultExportPath
#include "meshing/processing_modifiers/processing_mesh_utils.h" // EndsWithNoCase, ExportDialogStartPath
#include "ui/imgui_widgets.h" // ImGui::AssetSlot, ImGui::InputText

#include <picojson/picojson.h>

#include <superdex_robotics/utils/file_utils.h> // kRenderSubdir, EnsureDirectoriesCreated

#include <imguios/imguios.h>

#include <mochi_core/utils/dynamic_string.h>

#include <mochi_mesh/step_visual_export.h> // ExportStepVisual (visual-mesh path)

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace superdex::studio {

// Output format + tessellation backend + edge sampling, reflected as string names for readable
// JSON. The enumerator order matches the mochi wire types (VisualMeshFormat, CadMeshingBackend,
// StepMeshBodyParams::EdgeSampling) so props convert with a plain static_cast.
enum class VisualMeshFormat { Glb, Gltf, Obj, Stl };
enum class VisualMeshBackend { Delabella, Isotropic };
enum class VisualEdgeSampling { Uniform, Adaptive };

} // namespace superdex::studio

// Enum reflection specializes the global SReflectTypeTraits template, so it must be declared at
// global scope (the fully-qualified names refer back into superdex::studio).
MOCHI_ENUM_BEGIN(superdex::studio::VisualMeshFormat)
MOCHI_ENUM_ITEM(Glb)
MOCHI_ENUM_ITEM(Gltf)
MOCHI_ENUM_ITEM(Obj)
MOCHI_ENUM_ITEM(Stl)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::studio::VisualMeshBackend)
MOCHI_ENUM_ITEM(Delabella)
MOCHI_ENUM_ITEM(Isotropic)
MOCHI_ENUM_END()

MOCHI_ENUM_BEGIN(superdex::studio::VisualEdgeSampling)
MOCHI_ENUM_ITEM(Uniform)
MOCHI_ENUM_ITEM(Adaptive)
MOCHI_ENUM_END()

namespace superdex::studio {

namespace {

// Tessellation and output settings shared by both methods, mirroring StepVisualExportParams
// (whose defaults these match).
struct StepVisualParams {
  VisualMeshBackend backend = VisualMeshBackend::Isotropic;
  double linearDeflection = 0.1; // mm
  double angularDeflection = 0.5; // rad
  double targetEdgeLength = 0.0; // mm, 0 = auto
  double targetEdgeLengthFraction = 0.02; // fraction of bbox diagonal
  VisualEdgeSampling edgeSampling = VisualEdgeSampling::Adaptive;
  double scale = 0.001; // multiplier on the mm-normalized STEP geometry (0.001 = meters)
  bool allowPartialFailure = true;
  bool rgbaMaterialNames = false;

  MOCHI_STRUCT_BEGIN(superdex::studio::StepVisualParams)
  MOCHI_FIELD(backend)
  MOCHI_FIELD(linearDeflection)
  MOCHI_FIELD(angularDeflection)
  MOCHI_FIELD(targetEdgeLength)
  MOCHI_FIELD(targetEdgeLengthFraction)
  MOCHI_FIELD(edgeSampling)
  MOCHI_FIELD(scale)
  MOCHI_FIELD(allowPartialFailure)
  MOCHI_FIELD(rgbaMaterialNames)
  MOCHI_STRUCT_END()
};

// "From Model Viewer" parameters: shared visual settings plus the output path. The STEP itself
// comes from the editor's CAD slot at Run time (cached for SaveToFile).
struct ViewerVisualMeshProps {
  StepVisualParams params;
  mochi::DynamicString outputPath;

  MOCHI_STRUCT_BEGIN(superdex::studio::ViewerVisualMeshProps)
  MOCHI_FIELD(params)
  MOCHI_FIELD(outputPath)
  MOCHI_STRUCT_END()
};

// "From File" parameters: shared visual settings plus this modifier's own STEP slot and output
// path, independent of the editor's Model Viewer.
struct FileVisualMeshProps {
  StepVisualParams params;
  mochi::DynamicString stepPath;
  mochi::DynamicString outputPath;

  MOCHI_STRUCT_BEGIN(superdex::studio::FileVisualMeshProps)
  MOCHI_FIELD(params)
  MOCHI_FIELD(stepPath)
  MOCHI_FIELD(outputPath)
  MOCHI_STRUCT_END()
};

bool IsStepPath(std::string const& path) {
  return processing::EndsWithNoCase(path, ".step") || processing::EndsWithNoCase(path, ".stp");
}

mochi::mesh::VisualMeshFormat VisualFormatForPath(std::string const& path) {
  if (processing::EndsWithNoCase(path, ".gltf")) {
    return mochi::mesh::VisualMeshFormat::Gltf;
  }
  if (processing::EndsWithNoCase(path, ".obj")) {
    return mochi::mesh::VisualMeshFormat::Obj;
  }
  if (processing::EndsWithNoCase(path, ".stl")) {
    return mochi::mesh::VisualMeshFormat::Stl;
  }
  return mochi::mesh::VisualMeshFormat::Glb;
}

// The file a given path setting actually writes: the extension picks the format, and anything
// without a recognized one is exported as GLB with the suffix appended.
std::string NormalizeVisualExportPath(std::string path) {
  if (!path.empty() && !processing::EndsWithNoCase(path, ".glb") &&
      !processing::EndsWithNoCase(path, ".gltf") && !processing::EndsWithNoCase(path, ".obj") &&
      !processing::EndsWithNoCase(path, ".stl")) {
    path += ".glb";
  }
  return path;
}

// Converts one STEP file to one visual mesh file via the out-of-process helper, which writes the
// file itself preserving CAD colors and analytic normals. Shared by both methods' SaveToFile.
void WriteVisualMeshFile(
    std::string const& step,
    std::string const& path,
    StepVisualParams const& visual,
    mochi::Error& error) {
  MOCHI_ERROR_IF(step.empty(), error, "STEP to Visual Mesh: slot a STEP file first.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(
      !IsStepPath(step), error, "STEP to Visual Mesh: the slotted file is not a STEP file.");
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF(path.empty(), error, "STEP to Visual Mesh: set an output path first.");
  MOCHI_ERROR_RETURN(error);
  // The target folder may not exist yet -- a derived render/ folder in a fresh bot, or one the
  // user typed. Create it rather than failing the write.
  superdex::robotics::EnsureDirectoriesCreated(path, error);
  MOCHI_ERROR_RETURN(error);

  mochi::mesh::StepVisualExportParams params;
  params.backend = static_cast<mochi::mesh::CadMeshingBackend>(visual.backend);
  params.linearDeflection = visual.linearDeflection;
  params.angularDeflection = visual.angularDeflection;
  params.targetEdgeLength = visual.targetEdgeLength;
  params.targetEdgeLengthFraction = visual.targetEdgeLengthFraction;
  params.edgeSampling =
      static_cast<mochi::mesh::StepMeshBodyParams::EdgeSampling>(visual.edgeSampling);
  params.scale = visual.scale;
  params.allowPartialFailure = visual.allowPartialFailure;
  params.rgbaMaterialNames = visual.rgbaMaterialNames;

  mochi::mesh::VisualExportOutput output;
  output.format = VisualFormatForPath(path);
  output.path = path;
  std::vector<mochi::mesh::VisualExportOutput> const outputs{output};
  std::vector<mochi::mesh::VisualExportStatus> statuses;
  if (!mochi::mesh::ExportStepVisual(step, outputs, params, statuses, error)) {
    return;
  }
  // A true return means the STEP tessellated, but the single output may still have failed to
  // write (per-output failures do not fail the call), so check its status explicitly.
  MOCHI_ERROR_IF(
      statuses.size() != 1 || statuses.front() == mochi::mesh::VisualExportStatus::Failed,
      error,
      "STEP to Visual Mesh: the helper failed to write the visual mesh file.");
}

// Draws the STEP tessellation controls shared by both methods.
void ShowStepVisualParams(StepVisualParams& params, ModifierTooltip const& tooltip) {
  int backend = static_cast<int>(params.backend);
  char const* backendItems[] = {"Delabella (OCCT)", "Isotropic (CGAL)"};
  if (ImGui::Combo("Backend", &backend, backendItems, IM_ARRAYSIZE(backendItems))) {
    params.backend = static_cast<VisualMeshBackend>(backend);
  }
  tooltip(
      "Face tessellation backend:\n\n"
      "Delabella = OpenCascade's fast BRepMesh which aims for visual fidelity.\n\n"
      "Isotropic = Constrained Delaunay triangulation on uniformly sampled face points; "
      "highest quality for simple CAD with uniform edge length.");

  if (ImGui::InputDouble("Linear Deflection (mm)", &params.linearDeflection, 0.0, 0.0, "%.3f")) {
    params.linearDeflection = std::clamp(params.linearDeflection, 0.001, 10.0);
  }
  tooltip("Maximum chord deviation between the mesh and the true surface (mm). Smaller = finer.");

  if (ImGui::InputDouble("Angular Deflection (rad)", &params.angularDeflection, 0.0, 0.0, "%.3f")) {
    params.angularDeflection = std::clamp(params.angularDeflection, 0.01, 1.57);
  }
  tooltip("Maximum angle between adjacent facets (rad). Smaller = finer on curved surfaces.");

  // The remaining controls only affect the Isotropic (CGAL) path; Delabella meshes the whole
  // file using only the deflections above.
  if (params.backend == VisualMeshBackend::Isotropic) {
    int edgeSampling = static_cast<int>(params.edgeSampling);
    char const* edgeSamplingItems[] = {"Uniform", "Adaptive"};
    if (ImGui::Combo(
            "Edge Sampling", &edgeSampling, edgeSamplingItems, IM_ARRAYSIZE(edgeSamplingItems))) {
      params.edgeSampling = static_cast<VisualEdgeSampling>(edgeSampling);
    }
    tooltip(
        "How face boundary curves are sampled before triangulation. Adaptive respects angular "
        "deflection at the cost of shorter edge lengths along curves; better for visual meshes.");

    if (ImGui::InputDouble(
            "Target Edge Length mm (0=auto)", &params.targetEdgeLength, 0.0, 0.0, "%.4f")) {
      params.targetEdgeLength = std::max(params.targetEdgeLength, 0.0);
    }
    tooltip("Desired triangle edge length (mm). 0 = auto from the bounding-box fraction below.");

    ImGui::BeginDisabled(params.targetEdgeLength != 0.0);
    if (ImGui::InputDouble(
            "Edge Length Fraction", &params.targetEdgeLengthFraction, 0.0, 0.0, "%.4f")) {
      params.targetEdgeLengthFraction = std::clamp(params.targetEdgeLengthFraction, 0.001, 0.5);
    }
    tooltip(
        "Edge length is determined from the fraction of diagonal length of bounding-box of "
        "the input geometry.");
    ImGui::EndDisabled();

    ImGui::Checkbox("Allow Partial Failure", &params.allowPartialFailure);
    tooltip("Write a partial file when some faces fail to tessellate, instead of aborting.");
  }

  if (ImGui::InputDouble("Scale", &params.scale, 0.0, 0.0, "%.6f")) {
    params.scale = std::clamp(params.scale, 1e-6, 1.0);
  }
  tooltip(
      "Multiplier applied to the millimetre-normalized STEP geometry. 0.001 converts to "
      "metres; 1.0 keeps millimetres. The editor's CAD transform is not applied.");

  ImGui::Checkbox("RGBA Material Names", &params.rgbaMaterialNames);
  tooltip(
      "Rename each glTF material to the RRGGBBAA hex of its base color. glTF/GLB only; OBJ "
      "and STL ignore this.");
}

// The path Browse offers, and what RefreshAutoExportPath keeps the field at while Auto is on.
std::string SuggestedVisualExportPath(std::string const& stepFilePath) {
  return processing::DefaultExportPath(stepFilePath, superdex::robotics::kRenderSubdir, ".glb");
}

// Shared Auto/Browse output-path row. Toggling Auto back on re-derives via @p refreshAuto (each
// method derives from its own STEP source); Browse offers @p browseSuggestion.
void ShowOutputPathRow(
    mochi::DynamicString& outputPath,
    bool& autoPath,
    std::function<void()> const& refreshAuto,
    std::string const& browseSuggestion,
    ModifierGuiContext const& gui,
    char const* autoId,
    char const* browseId) {
  ModifierTooltip const& tooltip = gui.tooltip;

  ImGui::SeparatorText("Visual Mesh Export");
  ImGui::TextUnformatted("Auto");
  ImGui::SameLine();
  if (ImGui::Checkbox(autoId, &autoPath)) {
    refreshAuto(); // takes effect immediately when switched back on
  }
  tooltip(
      "Keep the output path derived from the STEP file, and leave it out of the saved "
      "pipeline. Turn this off (or press Browse) to choose the path yourself.");
  ImGui::SameLine();
  ImGui::BeginDisabled(autoPath);
  ImGui::InputText("Path", &outputPath);
  tooltip(
      "Output path; the extension selects the format (.glb, .gltf, .obj or .stl). No/unknown "
      "suffix defaults to .glb.");
  ImGui::EndDisabled();
  // Browse stays live while Auto is on: choosing a file IS taking the path over, so it turns
  // Auto off. Cancelling leaves everything as it was.
  if (ImGui::Button(browseId)) {
    char const* filters[] = {"*.glb", "*.gltf", "*.obj", "*.stl"};
    std::string const defaultPath = processing::ExportDialogStartPath(
        std::string{outputPath.data(), outputPath.size()},
        SuggestedVisualExportPath(browseSuggestion),
        gui.modelFolder);
    // The suggested folder may not exist yet (the asset uses role folders but has no
    // `render/`); create it up front, because the native dialog silently falls back to its
    // last-used directory when a default's parent is missing.
    superdex::robotics::EnsureDirectoriesCreated(
        defaultPath, mochi::ErrorLog{mochi::LogChannel::Warning});
    mochi::Path const chosen = SuperDexStudio::GetFileDialogPath(
        "Export Visual Mesh",
        filters,
        4,
        "Mesh (*.glb, *.gltf, *.obj, *.stl)",
        /*isSaveDialog=*/true,
        mochi::Path{defaultPath});
    if (!chosen.IsEmpty()) {
      outputPath = NormalizeVisualExportPath(chosen.ToString());
      autoPath = false;
    }
  }
  tooltip("Choose the output file (the extension picks the format).");
  if (gui.exportPathCollides) {
    ImGui::TextColored(
        ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "Another export writes to this same file.");
    tooltip(
        "Two enabled export modifiers share an output path, so one silently overwrites the "
        "other. Turn Auto off on one of them and give it a path of its own.");
  }
}

// Size of the visual mesh file on disk, when the stage has exported one. The editor composites
// this into the display string.
void AnnotateExportFileSize(std::string const& path, MeshStats& stats) {
  if (path.empty()) {
    return;
  }
  std::error_code ec;
  std::uintmax_t const size = std::filesystem::file_size(path, ec);
  if (!ec) {
    stats.fileSizeBytes = static_cast<int64_t>(size);
    stats.fileSizeLabel = "Export Size";
  }
}

// An auto path is reproduced on load, so it is left out of the document entirely; the presence of
// the key is exactly what marks a path as the user's own.
void SerializeWithoutAutoPath(
    ReflectedMethod<ViewerVisualMeshProps> const& method, picojson::value& out, bool autoPath);
void SerializeWithoutAutoPath(
    ReflectedMethod<FileVisualMeshProps> const& method, picojson::value& out, bool autoPath);

void SerializeWithoutAutoPath(
    ReflectedMethod<ViewerVisualMeshProps> const& method, picojson::value& out, bool autoPath) {
  method.ReflectedMethod<ViewerVisualMeshProps>::SerializeProps(out);
  if (autoPath && out.is<picojson::object>()) {
    out.get<picojson::object>().erase("outputPath");
  }
}

void SerializeWithoutAutoPath(
    ReflectedMethod<FileVisualMeshProps> const& method, picojson::value& out, bool autoPath) {
  method.ReflectedMethod<FileVisualMeshProps>::SerializeProps(out);
  if (autoPath && out.is<picojson::object>()) {
    out.get<picojson::object>().erase("outputPath");
  }
}

// ---- From Model Viewer --------------------------------------------------------------------------
class VisualMeshFromViewerMethod : public ReflectedMethod<ViewerVisualMeshProps> {
 public:
  char const* Name() const override {
    return "From Model Viewer";
  }
  char const* Description() const override {
    return "Uses the editor's current CAD Model (its Model Viewer slot) as the STEP source.";
  }
  bool CanGenerate(ModifierRunContext const& ctx) const override {
    return !ctx.cadFilePath.empty();
  }
  bool ProvidesFileExport() const override {
    return true;
  }
  std::string ExportPath() const override {
    return NormalizeVisualExportPath(
        std::string{_props.outputPath.data(), _props.outputPath.size()});
  }
  std::vector<std::string_view> PathPropKeys() const override {
    return {"outputPath"};
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) override {
    if (_autoExportPath) {
      _props.outputPath = SuggestedVisualExportPath(sourceFilePath);
    }
  }
  void SerializeProps(picojson::value& out) const override {
    SerializeWithoutAutoPath(*this, out, _autoExportPath);
  }
  void DeserializeProps(picojson::value const& in) override {
    ReflectedMethod::DeserializeProps(in);
    // A stored path means the user chose it, even if it happens to equal the current suggestion.
    _autoExportPath = _props.outputPath.empty();
  }
  // Fold the slotted file into the signature so swapping the CAD model re-runs this stage (which
  // refreshes the cached STEP path below).
  std::string PropsSignature(ModifierRunContext const& ctx) const override {
    std::string sig = SReflect::ToJsonString(_props, /*pretty=*/false);
    sig += '|';
    sig += ctx.cadFilePath;
    return sig;
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    ImGui::TextUnformatted("Uses the editor's current CAD Model as the STEP source.");
    if (gui.cadFilePath.empty()) {
      ImGui::TextDisabled("Slot a CAD Model (STEP) in the Model Viewer.");
      return;
    }
    if (!IsStepPath(gui.cadFilePath)) {
      ImGui::TextColored(
          ImVec4(1.0f, 0.4f, 0.3f, 1.0f),
          "The slotted file is not a STEP; visual export needs .step/.stp.");
      return;
    }
    ShowStepVisualParams(_props.params, gui.tooltip);
    ShowOutputPathRow(
        _props.outputPath,
        _autoExportPath,
        [this, &gui] { RefreshAutoExportPath(gui.sourceFilePath); },
        gui.cadFilePath,
        gui,
        "##autovisualmeshviewerpath",
        "Browse##visualmeshviewerbrowse");
  }

  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& ctx,
      mochi::Error& /*error*/) const override {
    // Cache the editor slot for SaveToFile (which receives no context). Export always cascades
    // Generate first, so this is current. Deliberately infallible: an empty slot only fails the
    // export itself, never the mesh chain passing through.
    _lastStepPath = ctx.cadFilePath;
    return {input}; // passthrough; the file is written by SaveToFile from the STEP slot
  }

  void SaveToFile(mochi::MeshData const& /*input*/, mochi::Error& error) const override {
    MOCHI_ERROR_IF(
        _lastStepPath.empty(),
        error,
        "STEP to Visual Mesh: no CAD file is slotted (generate the stage first).");
    MOCHI_ERROR_RETURN(error);
    WriteVisualMeshFile(_lastStepPath, ExportPath(), _props.params, error);
  }

  void AnnotateStats(MeshStats& stats) const override {
    AnnotateExportFileSize(ExportPath(), stats);
  }

 private:
  // Whether the export path is still derived from the STEP file rather than chosen by the user.
  // Not serialized: the presence of a stored path is what encodes it (see De/SerializeProps), so
  // a pipeline exporting to the derived location carries no path at all and stays valid when
  // copied into processing_presets/ or onto a machine whose assets live elsewhere. Runtime only.
  bool _autoExportPath = true;
  // Editor CAD slot captured by the last Run, consumed by SaveToFile (runtime only).
  mutable std::string _lastStepPath;
};

// ---- From File ----------------------------------------------------------------------------------
class VisualMeshFromFileMethod : public ReflectedMethod<FileVisualMeshProps> {
 public:
  char const* Name() const override {
    return "From File";
  }
  char const* Description() const override {
    return "Uses a STEP file from this modifier's own slot, independent of the editor's Model "
           "Viewer.";
  }
  bool CanGenerate(ModifierRunContext const& /*ctx*/) const override {
    return !_props.stepPath.empty();
  }
  bool ProvidesFileExport() const override {
    return true;
  }
  std::string ExportPath() const override {
    return NormalizeVisualExportPath(
        std::string{_props.outputPath.data(), _props.outputPath.size()});
  }
  std::vector<std::string_view> PathPropKeys() const override {
    return {"stepPath", "outputPath"};
  }
  void RefreshAutoExportPath(std::string const& sourceFilePath) override {
    if (!_autoExportPath) {
      return;
    }
    // The output is a visual mesh of this modifier's own STEP slot, so the suggestion derives
    // from that slot; fall back to the stack source when the slot is empty.
    std::string const step{_props.stepPath.data(), _props.stepPath.size()};
    _props.outputPath = SuggestedVisualExportPath(!step.empty() ? step : sourceFilePath);
  }
  void SerializeProps(picojson::value& out) const override {
    SerializeWithoutAutoPath(*this, out, _autoExportPath);
  }
  void DeserializeProps(picojson::value const& in) override {
    ReflectedMethod::DeserializeProps(in);
    // A stored path means the user chose it, even if it happens to equal the current suggestion.
    _autoExportPath = _props.outputPath.empty();
  }

  void ShowParams(ModifierGuiContext const& gui) override {
    if (gui.studio == nullptr || gui.assetManager == nullptr) {
      return;
    }
    ImGui::AssetSlot(
        "##visualmeshstepslot",
        _props.stepPath,
        *gui.assetManager,
        gui.studio,
        AssetType::CadModel,
        /*acceptDragDropPayload=*/true);
    gui.tooltip(
        "Drop a STEP file here (a .step/.stp in the 'cad' folder). This slot is independent of "
        "the editor's own Model Viewer slots. The stack mesh passes through untouched; the file "
        "below is written from the STEP, not from it.");
    std::string const step{_props.stepPath.data(), _props.stepPath.size()};
    if (!step.empty() && !IsStepPath(step)) {
      ImGui::TextColored(
          ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "Not a STEP file; visual export needs .step/.stp.");
      return;
    }
    if (step.empty()) {
      return;
    }
    ShowStepVisualParams(_props.params, gui.tooltip);
    ShowOutputPathRow(
        _props.outputPath,
        _autoExportPath,
        [this, &gui] { RefreshAutoExportPath(gui.sourceFilePath); },
        step,
        gui,
        "##autovisualmeshfilepath",
        "Browse##visualmeshfilebrowse");
  }

  mochi::MeshData Run(
      mochi::MeshData const& input,
      ModifierRunContext const& /*ctx*/,
      mochi::Error& /*error*/) const override {
    return {input}; // passthrough; the file is written by SaveToFile from the STEP slot
  }

  void SaveToFile(mochi::MeshData const& /*input*/, mochi::Error& error) const override {
    WriteVisualMeshFile(
        std::string{_props.stepPath.data(), _props.stepPath.size()},
        ExportPath(),
        _props.params,
        error);
  }

  void AnnotateStats(MeshStats& stats) const override {
    AnnotateExportFileSize(ExportPath(), stats);
  }

 private:
  // Whether the export path is still derived from the STEP file rather than chosen by the user.
  // Not serialized: the presence of a stored path is what encodes it (see De/SerializeProps), so
  // a pipeline exporting to the derived location carries no path at all and stays valid when
  // copied into processing_presets/ or onto a machine whose assets live elsewhere. Runtime only.
  bool _autoExportPath = true;
};

} // namespace

ModifierRegistryEntry MakeStepToVisualMeshEntry() {
  return ModifierRegistryEntry{"STEP to Visual Mesh", ModifierKind::Export, []() {
                                 std::vector<std::unique_ptr<MeshProcessingMethod>> methods;
                                 methods.push_back(
                                     std::make_unique<VisualMeshFromViewerMethod>());
                                 methods.push_back(std::make_unique<VisualMeshFromFileMethod>());
                                 return methods;
                               }};
}

} // namespace superdex::studio
